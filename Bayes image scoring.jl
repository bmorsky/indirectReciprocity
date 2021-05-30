using DifferentialEquations, Plots
#Probabilities that the donor is good given observations. Oij is the probability
#that the donor is good given the observation of a recipient doing action
#i={c,d}={cooperate,defect} to a recipient with reputation j={g,b}={good,bad}.
#Specifically, Ocg=P(good|→G), Ocb=P(good|→B), Odg=P(G|↛G), and Odb=P(G|↛B).

output = zeros(101^2,4)

#Parameters
e₁ = 0.01
e₂ = 0.01
ϵ = (1-e₁)*(1-e₂) + e₁*e₂
e = e₂
r = 3
τ = 10000
tspan = (0.0,1000)

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
                                        prob = ODEProblem(imgscore!,u₀,tspan)
                                        sol = solve(prob)
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
plot(output[:,1],output[:,2],group=output[:,3],seriestype = :scatter,xlims=(0,1),ylims=(0,1))
plot!(output[:,1][output[:,3].==1],output[:,2][output[:,3].==1],seriestype = :scatter,xlims=(0,1),ylims=(0,1))
plot!(output[:,1][output[:,3].==0],output[:,2][output[:,3].==0],seriestype = :scatter,xlims=(0,1),ylims=(0,1))


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
                                        gx=0.5
                                        gy=0.5
                                        gz=0.5
                                        x=0.06
                                        y=0.57
                                        z=0.37
                                        u₀ = [x;y;z;gx;gy;gz]
                                        prob = ODEProblem(imgscore!,u₀,(0.0,100))
                                        sol = solve(prob)
                                        plot(sol)
                                        plot!(sol[1,:],sol[2,:],xlims=(0,1),ylims=(0,1),arrow=true,linewidth = 2,legend=false)
                #                 end
                #         end
                # end
        end
end
current()

using PyCall, PyPlot

ternary = pyimport("ternary")

## Boundary and Gridlines
figure, tax = ternary.figure(scale=1.0)

# Draw Boundary and Gridlines
tax.boundary(linewidth=2.0)
tax.right_corner_label("AllC, x", fontsize=12)
tax.top_corner_label("Disc, z", fontsize=12)
tax.left_corner_label("AllD, y", fontsize=12)
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
tax.scatter(out[:,1:3][out[:,4].==0,:], linewidth=1.0)
tax.show()

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

tax.plot(EQ1, linewidth=2.0, color="black")
tax.clear_matplotlib_ticks()
tax.show()
gcf()
clf()

#output_e₃[:,1:3][output_e₃[:,4].==0,:]
