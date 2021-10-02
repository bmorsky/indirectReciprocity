using DifferentialEquations, Plots
#Probabilities that the donor is good given observations. Oij is the probability
#that the donor is good given the observation of a recipient doing action
#i={c,d}={cooperate,defect} to a recipient with reputation j={g,b}={good,bad}.
#Specifically, Ocg=P(good|→G), Ocb=P(good|→B), Odg=P(G|↛G), and Odb=P(G|↛B).

#Parameters
e₁ = 0.01
e₂ = 0.01
ϵ = (1-e₁)*(1-e₂) + e₁*e₂
e = e₂
r = 3
τ = 100
tspan = (0.0,2000)

function imgscore!(du,u,p,t)
        g = u[1]*u[4] + u[2]*u[5] + u[3]*u[6]

        #Imitation dynamics: u[1]-u[3]
        Px = r*(u[1] + u[3]*u[4]) - 1
        Py = r*(u[1] + u[3]*u[5])
        Pz = r*(u[1] + u[3]*u[6]) - g
        P̄ = u[1]*Px + u[2]*Py + u[3]*Pz
        du[1] = u[1]*(Px - P̄)
        du[2] = u[2]*(Py - P̄)
        du[3] = u[3]*(Pz - P̄)

        # Reputation dynamics: u[4]-u[6]
        Ocg = ϵ*g/(ϵ*g + e*(1 - g))
        Odg = (1 - ϵ)*g/((1 - ϵ)*g + (1 - e)*(1 - g))
        Icg = ϵ*Ocg + (1 - ϵ)*Odg
        Idg = (1 - e)*Odg + e*Ocg
        gx₊ = (1 - u[4])*Icg
        gx₋ = u[4]*(1 - Icg)
        gy₊ = (1 - u[5])*Idg
        gy₋ = u[5]*(1 - Idg)
        gz₊ = (1 - u[6])*(Icg*g + Idg*(u[1]*(1 - u[4]) + u[2]*(1 - u[5]) + u[3]*(1 - u[6])))
        gz₋ = u[6]*((1 - Icg)*g + (1 - Idg)*(u[1]*(1 - u[4]) + u[2]*(1 - u[5]) + u[3]*(1 - u[6])))
        du[4] = τ*(gx₊ - gx₋)
        du[5] = τ*(gy₊ - gy₋)
        du[6] = τ*(gz₊ - gz₋)
end

output = zeros(51^2,4)
count = 1
for m = 0:1:50
        x = m/50
        for n = 0:1:50-m
                y = n/50
                z = 1-x-y
                # for gx = 0:0.25:1
                #         for gy = 0:0.25:1
                #                 for gz = 0:0.25:1
                                        # further initial conditions
                                        x=0.5#rand()
                                        y=0.3#rand()
                                        z=0.2#rand()
                                        divisor=x+y+z
                                        x=x/divisor
                                        y=y/divisor
                                        z=z/divisor
                                        # x=0.2#1/3#rand()
                                        # y=0.3#1/3#rand()
                                        # z=0.5#1/3#rand()
                                        gx=0.5
                                        gy=0.5
                                        gz=0.5
                                        u₀ = [x;y;z;gx;gy;gz]
                                        prob = ODEProblem(imgscore!,u₀,tspan)
                                        sol = solve(prob)
                                        sol[:,end]
                                        plot(sol)
                                        output[count,1:3] = sol[1:3,end]
                                        if all(abs.(u₀[1:3] .- sol[1:3,end]) .< 0.01)
                                                global output[count,4] = 1
                                        end
                                        global count += 1
                #                 end
                #         end
                # end
        end
end
numEq = output
numEq = hcat(numEq[:,1],numEq[:,3],numEq[:,2],numEq[:,4])

current()

# Plot ternary figure
ternary = pyimport("ternary")
# Boundary and gridlines
figure, tax = ternary.figure(scale=1.0)
# Draw boundary and gridlines
tax.boundary(linewidth=2.0)
tax.right_corner_label(" AllC", fontsize=20)
tax.top_corner_label("Disc", fontsize=20)
tax.left_corner_label("AllD ", fontsize=20)
tax.get_axes().axis("off")
#numEq[:,1:3][numEq[:,4].==0,:]

numEq_stable = numEq[:,1:3][numEq[:,4].==0,:]
numEq_unstable = numEq[:,1:3][numEq[:,4].==1,:]

tax.scatter(numEq_stable, linewidths=10, color="blue", marker="o")
tax.show()
gcf()
tax.savefig("test_scoring_ternary")
clf()





plot(output[:,1],output[:,2],group=output[:,3],seriestype = :scatter,xlims=(0,1),ylims=(0,1))
plot!(output[:,1][output[:,3].==1],output[:,2][output[:,3].==1],seriestype = :scatter,xlims=(0,1),ylims=(0,1))
plot!(output[:,1][output[:,3].==0],output[:,2][output[:,3].==0],seriestype = :scatter,xlims=(0,1),ylims=(0,1))

using Plots
plot()
for m = 0:5:100
        x = m/100
        for n = 0:5:100-m
                y = n/100
                z = 1-x-y
                # for gx = 0:0.25:1
                #         for gy = 0:0.25:1
                #                 for gz = 0:0.25:1
                                        # further initial conditions
                                        gx=rand()
                                        gy=rand()
                                        gz=rand()
                                        x=rand()
                                        y=rand()
                                        z=rand()

                                        function σ_f(du,u,p,t)
                                          du[1] = 0.0001*u[1]
                                          du[2] = 0.0001*u[2]
                                          du[3] = 0.0001*u[3]
                                          du[4] = 0.001*u[4]
                                          du[5] = 0.001*u[5]
                                          du[6] = 0.001*u[6]
                                        end


                                        divisor = x+y+z
                                        x/divisor
                                        y/divisor
                                        z/divisor
                                        u₀ = [x/divisor;y/divisor;z/divisor;gx;gy;gz]

                                        prob = ODEProblem(imgscore!,u₀,(0.0,5000))
                                        sol = solve(prob)
                                        sol[:,end]
                                        Plots.plot(sol)
                                        Plots.plot(sol,vars = [(0,1), (0,2), (0,3)])
                                        plot!(sol[1,:],sol[2,:],xlims=(0,1),ylims=(0,1),arrow=true,linewidth = 2,legend=false)
                #                 end
                #         end
                # end
        end
end
current()

using PyCall, PyPlot

out = hcat(output[:,1],output[:,3],output[:,2],output[:,4])
tax.scatter(out[:,1:3], linewidth=1.0)
tax.show()

gcf()

ternary = pyimport("ternary")

## Boundary and Gridlines
figure, tax = ternary.figure(scale=1.0)

# Draw Boundary and Gridlines
tax.boundary(linewidth=2.0)
tax.right_corner_label(" AllC", fontsize=20)
tax.top_corner_label("Disc", fontsize=20)
tax.left_corner_label("AllD ", fontsize=20)
tax.get_axes().axis("off")
# Set Axis labels and Title

# Remove default Matplotlib Axes
tax.clear_matplotlib_ticks()
matplotlib.pyplot.box(False)
ternary.plt.show()
#ternary.plt.savefig("mygraph.png")

## Sample trajectory plot
figure, tax = ternary.figure(scale=1.0)
tax.boundary()
# Load some data, tuples (x,y,z)
# with open("sample_data/curve.txt") as handle:
#     for line in handle:
#         points.append(list(map(float, line.split(' '))))
# Plot the data
tax.plot(sol[1:3,:]', linewidth=2.0)
tax.clear_matplotlib_ticks()
tax.show()

gcf()
clf()
out = hcat(output_e₃[:,1],output_e₃[:,3],output_e₃[:,2],output_e₃[:,4])

out1 = out[:,1:3][out[:,4].==0,:]
out1 = out1[out1[:,1].<0.01,:]
out1 = out1[out1[:,2].<0.8,:]

out2 = out[:,1:3][out[:,4].==0,:]
out2 = out2[out2[:,1].>0.01,:]
out2 = vcat(out2,[0 1 0])
out2 = out2[sortperm(out2[:, 3]), :]

tax.plot(out2, linewidth=7, color="orangered", solid_capstyle="round")

tax.show()
gcf()
tax.savefig("imagescore_ternary")
gcf()


## Third type of error case
e₃ = e
output_e₃ = zeros(101^2,4)

function imgscore_e₃!(du,u,p,t)
        g = u[1]*u[4] + u[2]*u[5] + u[3]*u[6]

        #Imitation dynamics: u[1]-u[3]
        Px = r*(u[1] + u[3]*u[4]) - 1
        Py = r*(u[1] + u[3]*u[5])
        Pz = r*(u[1] + u[3]*u[6]) - g
        P̄ = u[1]*Px + u[2]*Py + u[3]*Pz
        du[1] = u[1]*(Px - P̄)
        du[2] = u[2]*(Py - P̄)
        du[3] = u[3]*(Pz - P̄)

        # Reputation dynamics: u[4]-u[6]
        Ocg = ϵ*g/(ϵ*g + e*(1 - g))
        Odg = (1 - ϵ)*g/((1 - ϵ)*g + (1 - e)*(1 - g))
        Icg = (1-e₃)*(ϵ*Ocg + (1 - ϵ)*Odg) + e₃*(ϵ*(1 - Ocg) + (1 - ϵ)*(1 - Odg))
        Idg = (1-e₃)*((1 - e)*Odg + e*Ocg) + e₃*((1 - e)*(1 - Odg) + e*(1 - Ocg))
        gx₊ = (1 - u[4])*Icg
        gx₋ = u[4]*(1 - Icg)
        gy₊ = (1 - u[5])*Idg
        gy₋ = u[5]*(1 - Idg)
        gz₊ = (1 - u[6])*(Icg*g + Idg*(u[1]*(1 - u[4]) + u[2]*(1 - u[5]) + u[3]*(1 - u[6])))
        gz₋ = u[6]*((1 - Icg)*g + (1 - Idg)*(u[1]*(1 - u[4]) + u[2]*(1 - u[5]) + u[3]*(1 - u[6])))
        du[4] = τ*(gx₊ - gx₋)
        du[5] = τ*(gy₊ - gy₋)
        du[6] = τ*(gz₊ - gz₋)
end

count = 1
for m = 0:1:100
        x = m/100
        for n = 0:1:100-m
                y = n/100
                z = 1-x-y
                # for gx = 0:0.25:1
                #         for gy = 0:0.25:1
                #                 for gz = 0:0.25:1
                                        # further initial conditions
                                        gx=0.5
                                        gy=0.5
                                        gz=0.5
                                        u₀ = [x;y;z;gx;gy;gz]
                                        prob = ODEProblem(imgscore_e₃!,u₀,tspan)
                                        sol = solve(prob)
                                        output_e₃[count,1:3] = sol[1:3,end]
                                        if all(abs.(u₀[1:3] .- sol[1:3,end]) .< 0.01)
                                                global output_e₃[count,4] = 1
                                        end
                                        global count += 1
                #                 end
                #         end
                # end
        end
end

tax.scatter([result.zero[1] 0.5 result.zero[2]], linewidth=2.0)
tax.clear_matplotlib_ticks()
tax.show()

gcf()
clf()


#output_e₃[:,1:3][output_e₃[:,4].==0,:]

## Calculate equiibria

using Calculus, ModelingToolkit, NLsolve

function f!(u)
        g = u[1]*u[3] + u[2]*u[4] + z*u[5]

        #Imitation dynamics: u[1]-u[3]
        Px = r*(u[1] + z*u[3]) - 1
        Py = r*(u[1] + z*u[4])
        Pz = r*(u[1] + z*u[5]) - g
        P̄ = u[1]*Px + u[2]*Py + z*Pz
        F[1] = u[1]*(Px - P̄)
        F[2] = u[2]*(Py - P̄)
        F[3] = z*(Pz - P̄)

        # Reputation dynamics: u[4]-u[6]
        Ocg = ϵ*g/(ϵ*g + e*(1 - g))
        Odg = (1 - ϵ)*g/((1 - ϵ)*g + (1 - e)*(1 - g))
        Icg = ϵ*Ocg + (1 - ϵ)*Odg
        Idg = (1 - e)*Odg + e*Ocg
        gx₊ = (1 - u[3])*Icg
        gx₋ = u[3]*(1 - Icg)
        gy₊ = (1 - u[4])*Idg
        gy₋ = u[4]*(1 - Idg)
        gz₊ = (1 - u[5])*(Icg*g + Idg*(u[1]*(1 - u[3]) + u[2]*(1 - u[4]) + z*(1 - u[5])))
        gz₋ = u[5]*((1 - Icg)*g + (1 - Idg)*(u[1]*(1 - u[3]) + u[2]*(1 - u[4]) + z*(1 - u[5])))
        F[4] = τ*(gx₊ - gx₋)
        F[5] = τ*(gy₊ - gy₋)
        F[6] = τ*(gz₊ - gz₋)
        return F
end

function j!(J,u)

end
@parameters t
@variables x(t) y(t)
@derivatives D'~t

Ll = [D(x)~x+y, D(y)~x^2]
de = ODESystem(Ll)
calculate_jacobian(de)

g(x) = x*g(x) + 1-x-z*gy + z*gz
Px(x) = r*(x + z*gx) - 1
Py(x) = r*(x + z*gy)
Pz(x) = r*(x + z*gz) - g
P̄ = u[1]*Px + u[2]*Py + u[3]*Pz
eq = x*(Px - P̄)

initial_conditions = [0.5; 0.5; 0.5; 0.5; 0.5; 0.5]

EQ = zeros(1001,3)
for m = 1:1:1001
        z=(m-1)/1000
        function ww!(F,u)
                g = u[1]*u[3] + u[2]*u[4] + z*u[5]

                #Imitation dynamics: u[1]-u[3]
                Px = r*(u[1] + z*u[3]) - 1
                Py = r*(u[1] + z*u[4])
                Pz = r*(u[1] + z*u[5]) - g
                P̄ = u[1]*Px + u[2]*Py + z*Pz

                # Reputation dynamics: u[4]-u[6]
                Ocg = ϵ*g/(ϵ*g + e*(1 - g))
                Odg = (1 - ϵ)*g/((1 - ϵ)*g + (1 - e)*(1 - g))
                Icg = ϵ*Ocg + (1 - ϵ)*Odg
                Idg = (1 - e)*Odg + e*Ocg
                gx₊ = (1 - u[3])*Icg
                gx₋ = u[3]*(1 - Icg)
                gy₊ = (1 - u[4])*Idg
                gy₋ = u[4]*(1 - Idg)
                gz₊ = (1 - u[5])*(Icg*g + Idg*(u[1]*(1 - u[3]) + u[2]*(1 - u[4]) + z*(1 - u[5])))
                gz₋ = u[5]*((1 - Icg)*g + (1 - Idg)*(u[1]*(1 - u[3]) + u[2]*(1 - u[4]) + z*(1 - u[5])))

                F[1] = u[1]*(Px - P̄);
                F[2] = u[2]*(Py - P̄);
                F[3] = τ*(gx₊ - gx₋);
                F[4] = τ*(gy₊ - gy₋);
                F[5] = τ*(gz₊ - gz₋)
        end
        result = nlsolve(ww!, [(1-z)/2; (1-z)/2; 0.5; 0.5; 0.5])
        if 1 .> sum(result.zero[1:2])+z .>= 0
                EQ[m,:] = [result.zero[1] z result.zero[2]]
        end
end

EQ1 = [0 0 0]
for m=1:1:1001
        if all(EQ[m,:] .!= 0)
                EQ1 = vcat(EQ1,EQ[m,:]')
        end
end

tax.scatter(EQ1, linewidth=1.0, color="black")
tax.clear_matplotlib_ticks()
tax.show()
gcf()
clf()

#output_e₃[:,1:3][output_e₃[:,4].==0,:]



#output_e₃[:,1:3][output_e₃[:,4].==0,:]

## Calculate equiibria 2nd time

using NLsolve

EQ = zeros(1000,3)
for m = 1:1:1000
        x=m/1000
        function ww!(F,u)
                g = x*u[2] + (1-u[1]-x)*u[3] + u[1]*u[4]

                #Imitation dynamics: u[1]-u[3]
                Px = r*(x + u[1]*u[2]) - 1
                Py = r*(x + u[1]*u[3])
                Pz = r*(x + u[1]*u[4]) - g
                P̄ = x*Px + (1-u[1]-x)*Py + u[1]*Pz

                # Reputation dynamics: u[4]-u[6]
                Ocg = ϵ*g/(ϵ*g + e*(1 - g))
                Odg = (1 - ϵ)*g/((1 - ϵ)*g + (1 - e)*(1 - g))
                Icg = ϵ*Ocg + (1 - ϵ)*Odg
                Idg = (1 - e)*Odg + e*Ocg
                gx₊ = (1 - u[2])*Icg
                gx₋ = u[2]*(1 - Icg)
                gy₊ = (1 - u[3])*Idg
                gy₋ = u[3]*(1 - Idg)
                gz₊ = (1 - u[4])*(Icg*g + Idg*(1-g))
                gz₋ = u[4]*((1 - Icg)*g + (1 - Idg)*(1-g))

                F[1] = u[1]*(Pz - P̄)
                F[2] = (Icg - u[2])
                F[3] = (Idg - u[3])
                F[4] = (Icg*g + Idg*(1-g) - u[4])
                F[5] = x*(Px - P̄)
                F[6] = (1-u[1]-x)*(Py - P̄)
        end
        result = nlsolve(ww!, [(1-x)*0.5; 0.5; 0.5; 0.5; x; (1-x)*0.5])
        EQ[m,:] = [x result.zero[1] 1-result.zero[1]-x]
end

EQ1 = [1 0 0]
for m=1:1:1000
        if all(EQ[m,:] .!= 0)
                EQ1 = vcat(EQ1,EQ[m,:]')
        end
end
for m=1:1:size(EQ1,1)
        for n=1:3
                if EQ1[m,n] < 0
                        EQ1[m,n] = 0
                end
                if EQ1[m,n] > 1
                        EQ1[m,n] = 1
                end
        end
end

## Boundary and Gridlines
# clf()
figure, tax = ternary.figure(scale=1.0)

# Draw Boundary and Gridlines
tax.boundary(linewidth=2.0)
tax.right_corner_label("AllC, x", fontsize=12)
tax.top_corner_label("Disc, z", fontsize=12)
tax.left_corner_label("AllD, y", fontsize=12)
tax.get_axes().axis("off")

tax.scatter(EQ1, linewidth=1.0, color="black")
tax.show()
gcf()
tax.savefig("imagescore_ternary")

#output_e₃[:,1:3][output_e₃[:,4].==0,:]
