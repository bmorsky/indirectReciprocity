using DifferentialEquations, NLsolve, Plots, PyCall, PyPlot
# Probabilities that the donor is good given observations. Oij is the
# probability that the donor is good given the observation of then doing action
# i={c,d}={cooperate,defect} to a recipient with reputation j={g,b}={good,bad}.
# Specifically, Ocg=P(G|→G), Ocb=P(G|→B), Odg=P(G|↛G), and  Odb=P(G|↛B).

# Parameters
e₁ = 0.01
e₂ = 0.01
ϵ = (1-e₁)*(1-e₂) + e₁*e₂
e = e₂
r = 3
τ = 10000
T = 2000
tspan = (0.0,T)

function shunning!(du,u,p,t)
        g = u[1]*u[4] + u[2]*u[5] + u[3]*u[6]
        g2 = u[1]*u[4]^2 + u[2]*u[5]^2 + u[3]*u[6]^2
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
        Icb = 0 # ϵ*Ocb + (1 - ϵ)*Odb = 0, since Ocb = Odb = 0
        Idb = 0 # (1 - e)*Odb + e*Ocb = 0, since Ocb = Odb = 0
        gx₊ = (1 - u[4])*Icg*g
        gx₋ = u[4]*((1 - Icg)*g + 1-g)
        gx2₊ = (u[4] - u[7])*Icg*g
        gx2₋ = u[7]*((1 - Icg)*g + 1-g)
        gy₊ = (1 - u[5])*Idg*g
        gy₋ = u[5]*((1 - Idg)*g + 1-g)
        gy2₊ = (u[5] - u[8])*Idg*g
        gy2₋ = u[8]*((1 - Idg)*g + 1-g)
        gz₊ = (1 - u[6])*(Icg*g2 + (g-g2)*Idg)
        gz₋ = u[6]*((1-Icg)*g2 + (g-g2)*(1-Idg) + 1-g)
        gz2₊ = (u[6] - u[9])*(Icg*g2 + (g-g2)*Idg)
        gz2₋ = u[9]*((1-Icg)*g2 + (g-g2)*(1-Idg) + 1-g)
        du[4] = τ*(gx₊ - gx₋)
        du[5] = τ*(gy₊ - gy₋)
        du[6] = τ*(gz₊ - gz₋)
        du[7] = τ*(gx2₊ - gx2₋)
        du[8] = τ*(gy2₊ - gy2₋)
        du[9] = τ*(gz2₊ - gz2₋)
end

# Numerically solve for different initial conditions to find equilibria.
numsims = 50
numEq = zeros((numsims+1)^2,4)
count = 1
for m = 0:1:numsims
        x = m/numsims
        for n = 0:1:numsims-m
                y = n/numsims
                z = 1-x-y
                # for gx = 0:0.25:1
                #         for gy = 0:0.25:1
                #                 for gz = 0:0.25:1
                                        # further initial conditions
                                        gx=0.5
                                        gy=0.5
                                        gz=0.5
                                        gx2=0.25
                                        gy2=0.25
                                        gz2=0.25
                                        u₀ = [x;y;z;gx;gy;gz;gx2;gy2;gz2]
                                        prob = ODEProblem(shunning!,u₀,tspan)
                                        sol = solve(prob)
                                        numEq[count,1:3] = sol[1:3,end]
                                        if all(abs.(u₀[1:3] .- sol[1:3,end]) .< 0.01)
                                                global numEq[count,4] = 1
                                        end
                                        global count += 1
                #                 end
                #         end
                # end
        end
end
plot(numEq[:,1],numEq[:,2],group=numEq[:,3],seriestype = :scatter,xlims=(0,1),ylims=(0,1))
plot!(numEq[:,1][numEq[:,3].==1],numEq[:,2][numEq[:,3].==1],seriestype = :scatter,xlims=(0,1),ylims=(0,1))
plot!(numEq[:,1][numEq[:,3].==0],numEq[:,2][numEq[:,3].==0],seriestype = :scatter,xlims=(0,1),ylims=(0,1))
numEq = hcat(numEq[:,1],numEq[:,3],numEq[:,2],numEq[:,4])

# Plot paths in phase space.
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
                                        gx2=0.25
                                        gy2=0.25
                                        gz2=0.25
                                        x=0.1#rand()
                                        y=0.8#rand()
                                        z=0.1#rand()
                                        divisor = x+y+z
                                        u₀ = [x/divisor;y/divisor;z/divisor;gx;gy;gz;gx2;gy2;gz2]
                                        prob = ODEProblem(shunning!,u₀,(0.0,10000000.0))
                                        sol = solve(prob)
                                        plot(sol,vars = [(0,1), (0,2), (0,3)])
                                        plot(sol[1,:],sol[2,:],xlims=(0,1),ylims=(0,1),arrow=true,linewidth = 2,legend=false)
                #                 end
                #         end
                # end
        end
end
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
tax.scatter(numEq[:,1:3][numEq[:,4].==0,:], linewidths=10, color="orangered", marker="o")
tax.show()
gcf()
tax.savefig("shunning_ternary")
clf()

## Third type of error case
e₃ = e

function shunning_e₃!(du,u,p,t)
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

numEq_e₃ = zeros(501^2,4)
count = 1
for m = 0:1:500
        x = m/500
        for n = 0:1:500-m
                y = n/500
                z = 1-x-y
                # for gx = 0:0.25:1
                #         for gy = 0:0.25:1
                #                 for gz = 0:0.25:1
                                        # further initial conditions
                                        gx=0.5
                                        gy=0.5
                                        gz=0.5
                                        u₀ = [x;y;z;gx;gy;gz]
                                        prob = ODEProblem(shunning_e₃!,u₀,tspan)
                                        sol = solve(prob)
                                        numEq_e₃[count,1:3] = sol[1:3,end]
                                        if all(abs.(u₀[1:3] .- sol[1:3,end]) .< 0.01)
                                                global numEq_e₃[count,4] = 1
                                        end
                                        global count += 1
                #                 end
                #         end
                # end
        end
end
numEq_e₃ = hcat(numEq_e₃[:,1],numEq_e₃[:,3],numEq_e₃[:,2],numEq_e₃[:,4])

tax.scatter(numEq_e₃[:,1:3], linewidth=1.0)
tax.show()
gcf()
#clf()

rootsEqz = zeros(1001,3)
for m = 1:1:1001
        z=(m-1)/1000
        function f!(F,u)
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
        result = nlsolve(f!, [(1-z)/2; (1-z)/2; 0.5; 0.5; 0.5])
        if 1 .> sum(result.zero[1:2])+z .>= 0
                rootsEqz[m,:] = [result.zero[1] z result.zero[2]]
        end
end
rootsEqz = rootsEqz[vec(sum(rootsEqz,dims=2).!=0),:]
for m=1:1:size(rootsEqz,1)
        for n=1:3
                if rootsEqz[m,n] < 0
                        rootsEqz[m,n] = 0
                end
                if rootsEqz[m,n] > 1
                        rootsEqz[m,n] = 1
                end
        end
end

# Plot
tax.scatter(rootsEqz, linewidth=1.0, color="black")
tax.show()
gcf()
#clf()

# Calculate equilibria 2nd time given x
rootsEqx = zeros(1000,3)
for m = 1:1:1000
        x=m/1000
        function f!(F,u)
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
                gz₊ = (1 - u[4])*(Icg*g + Idg*(x*(1 - u[2]) + (1-u[1]-x)*(1 - u[3]) + u[1]*(1 - u[4])))
                gz₋ = u[4]*((1 - Icg)*g + (1 - Idg)*(x*(1 - u[2]) + (1-u[1]-x)*(1 - u[3]) + u[1]*(1 - u[4])))

                F[1] = u[1]*(Pz - P̄)
                F[2] = τ*(gx₊ - gx₋)
                F[3] = τ*(gy₊ - gy₋)
                F[4] = τ*(gz₊ - gz₋)
        end
        result = nlsolve(f!, [(1-x)/2; 0.5; 0.5; 0.5], method = :newton)
        rootsEqx[m,:] = [x result.zero[1] 1-result.zero[1]-x]
end
rootsEqx = rootsEqx[vec(sum(rootsEqx,dims=2).!=0),:]
for m=1:1:size(rootsEqx,1)
        for n=1:3
                if rootsEqx[m,n] < 0
                        rootsEqx[m,n] = 0
                end
                if rootsEqx[m,n] > 1
                        rootsEqx[m,n] = 1
                end
        end
end

# Plot
tax.scatter(rootsEqx, linewidth=1.0, color="black")
tax.show()
gcf()
#clf()
