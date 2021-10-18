using DifferentialEquations, Distributions, Plots, SpecialFunctions, Sundials

#Parameters
e₁ = 0.01
e₂ = 0.01
ϵ = (1-e₁)*(1-e₂) + e₁*e₂
e = e₂
r = 3
λ = 0.9
bias = 0
tspan = (0.0,1000)

function simplestanding!(du,u,p,t)
    g = p[1]*u[1] + p[2]*u[2] + (1-p[1]-p[2])*u[3]
    ĝ = -2*g^3 + 3*g^2#-λ*g^2 + 2*λ*g + 1 - λ #λ*g + (1-λ)*bias#min(g*1.1, 1.0)#
    g2 = p[1]*u[4] + p[2]*u[5] + (1-p[1]-p[2])*u[6]

    # Reputation dynamics: u[1]-u[6]
    Pcg = ϵ*ĝ/(ϵ*ĝ + e*(1 - ĝ))
    Pdg = (1 - ϵ)*ĝ/((1 - ϵ)*ĝ + (1 - e)*(1 - ĝ))
    Qcg = ϵ*Pcg + (1 - ϵ)*Pdg
    Qdg = (1 - e)*Pdg + e*Pcg
    Qcb = 1
    Qdb = 1
    gx₊ = (1 - u[1])*(Qcg*g + 1 - g)
    gx₋ = u[1]*(1-Qcg)*g
    gx2₊ = (u[1] - u[4])*(Qcg*g + 1 - g)
    gx2₋ = u[4]*(1-Qcg)*g
    gy₊ = (1 - u[2])*(Qdg*g + 1 - g)
    gy₋ = u[2]*(1-Qdg)*g
    gy2₊ = (u[2] - u[5])*(Qdg*g + 1 - g)
    gy2₋ = u[5]*(1-Qdg)*g
    gz₊ = (1 - u[3])*(Qcg*g2 + Qdg*(g-g2) + 1 - g)
    gz₋ = u[3]*((1-Qcg)*g2 + (1-Qdg)*(g-g2))
    gz2₊ = (u[3] - u[6])*(Qcg*g2 + Qdg*(g-g2) + 1 - g)
    gz2₋ = u[6]*((1-Qcg)*g2 + (1-Qdg)*(g-g2))
    du[1] = gx₊ - gx₋
    du[2] = gy₊ - gy₋
    du[3] = gz₊ - gz₋
    du[4] = gx2₊ - gx2₋
    du[5] = gy2₊ - gy2₋
    du[6] = gz2₊ - gz2₋
end

# The differentia-algebraic equation (DAE) we wish to solve
function DAE(out,du,u,p,t)
    g = u[1]*u[3] + u[2]*u[4] + (1-u[1]-u[2])*u[5]
    ĝ = -2*g^3 + 3*g^2#-λ*g^2 + 2*λ*g + 1 - λ #λ*g + (1-λ)*bias#min(g*1.1, 1.0)#
    g2 = u[1]*u[6] + u[2]*u[7] + (1-u[1]-u[2])*u[8]

    #Imitation dynamics: u[1]-u[3]
    Px = r*(u[1] + (1-u[1]-u[2])*u[3]) - 1
    Py = r*(u[1] + (1-u[1]-u[2])*u[4])
    Pz = r*(u[1] + (1-u[1]-u[2])*u[5]) - g
    P̄ = u[1]*Px + u[2]*Py + (1-u[1]-u[2])*Pz
    out[1] = u[1]*(Px - P̄) - du[1]
    out[2] = u[2]*(Py - P̄) - du[2]

    # Reputation dynamics: u[3]-u[8]
    Pcg = ϵ*ĝ/(ϵ*ĝ + e*(1 - ĝ))
    Pdg = (1 - ϵ)*ĝ/((1 - ϵ)*ĝ + (1 - e)*(1 - ĝ))
    Qcg = ϵ*Pcg + (1 - ϵ)*Pdg
    Qdg = (1 - e)*Pdg + e*Pcg
    Qcb = 1
    Qdb = 1
    gx₊ = (1 - u[3])*(Qcg*g + 1 - g)
    gx₋ = u[3]*(1-Qcg)*g
    gx2₊ = (u[3] - u[6])*(Qcg*g + 1 - g)
    gx2₋ = u[6]*(1-Qcg)*g
    gy₊ = (1 - u[4])*(Qdg*g + 1 - g)
    gy₋ = u[4]*(1-Qdg)*g
    gy2₊ = (u[4] - u[7])*(Qdg*g + 1 - g)
    gy2₋ = u[7]*(1-Qdg)*g
    gz₊ = (1 - u[5])*(Qcg*g2 + Qdg*(g-g2) + 1 - g)
    gz₋ = u[5]*((1-Qcg)*g2 + (1-Qdg)*(g-g2))
    gz2₊ = (u[5] - u[8])*(Qcg*g2 + Qdg*(g-g2) + 1 - g)
    gz2₋ = u[8]*((1-Qcg)*g2 + (1-Qdg)*(g-g2))
    out[3] = gx₊ - gx₋
    out[4] = gy₊ - gy₋
    out[5] = gz₊ - gz₋
    out[6] = gx2₊ - gx2₋
    out[7] = gy2₊ - gy2₋
    out[8] = gz2₊ - gz2₋
end

output = zeros(5151,7)#zeros(4930,7) #
outcontour = Array{Float64}(undef,101,101)
count = 1
for x = 0:0.01:1#0.01:0.01:0.99 #
    for y = 0:0.01:1-x#0.01:0.01:1-x #
        # Initial conditions for u
        prob = ODEProblem(simplestanding!,[0.5,0.5,0.5,0.25,0.25,0.25],tspan,[x,y])
        sol = solve(prob)
        (gx, gy, gz) = sol[:,end]
        u₀ = [x, y, gx, gy, gz, gx^2, gy^2, gz^2]
        # Initial condition for du
        g = x*gx + y*gy + (1-x-y)*gz
        ĝ = λ*g + 1 - λ
        Px = r*(x + (1-x-y)*gx) - 1
        Py = r*(x + (1-x-y)*gy)
        Pz = r*(x + (1-x-y)*gz) - g
        P̄ = x*Px + y*Py + (1-x-y)*Pz
        du₀ = [x*(Px - P̄), y*(Py - P̄), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # Solve DAE
        differential_vars = [true,true,false,false,false,false,false,false]
        prob = DAEProblem(DAE,du₀,u₀,tspan,differential_vars=differential_vars)
        sol = solve(prob,IDA(),linearsolver=:Dense)
        output[count,:] = hcat([x y],sol[1:5,end]')
        count += 1
        outcontour[round(Int,x*10)+1,round(Int,y*10)+1] = sol[1,end] .+ (1 .- sol[1,end] .- sol[2,end]).*(sol[1,end].*sol[3,end] + sol[2,end].*sol[4,end] .+ (1 .- sol[1,end] .- sol[2,end]).*sol[5,end])
    end
end

using PyCall, PyPlot
# current()

finalstates = hcat(output[:,3],1 .- output[:,3] .- output[:,4],output[:,4])
numEq = hcat(output[:,1],1 .- output[:,1] .- output[:,2],output[:,2],output[:,3])
cooperation = output[:,3] .+ (1 .- output[:,3] .- output[:,4]).*(output[:,3].*output[:,5] + output[:,4].*output[:,6] .+ (1 .- output[:,3] .- output[:,4]).*output[:,7])
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
numEq_stable = numEq[:,1:3][numEq[:,4].>=0.00001,:]
numEq_unstable = numEq[:,1:3][numEq[:,4].<=0.00001,:]
tax.scatter(finalstates, linewidths=0.01, marker="o")
tax.scatter(numEq[:,1:3], c = cooperation, linewidths=0.0001, marker=",")
tax.show()
gcf()
# tax.savefig("DAE_ternary")
# clf()
#
# plt = PyPlot.contourf(0:0.01:1, 0:0.01:1, outcontour; levels = collect(0:0.01:1))
# gcf()

# clf()


# x = 0.3
# y = 0.2
# prob = ODEProblem(simplestanding!,[0.5,0.5,0.5,0.25,0.25,0.25],(0.0,10000000),[x,y])
# sol = solve(prob)
# (gx, gy, gz) = sol[:,end]
# u₀ = [x, y, gx, gy, gz, gx^2, gy^2, gz^2]
# # Initial condition for du
# g = x*gx + y*gy + (1-x-y)*gz
# ĝ = λ*g + 1 - λ
# Px = r*(x + (1-x-y)*gx) - 1
# Py = r*(x + (1-x-y)*gy)
# Pz = r*(x + (1-x-y)*gz) - g
# P̄ = x*Px + y*Py + (1-x-y)*Pz
# du₀ = [x*(Px - P̄), y*(Py - P̄), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# # Solve DAE
# differential_vars = [true,true,false,false,false,false,false,false]
# prob = DAEProblem(DAE,du₀,u₀,(0.0,10000000),differential_vars=differential_vars)
# sol = solve(prob,IDA(),linearsolver=:Dense)
#
# pyplot = pyimport("matplotlib.pyplot")
# clf()
# pyplot.plot(sol.t, sol[1:5,:]',label=["x","y","gx","gy","gz"])
# pyplot.legend()
# gcf()
#
# (x,y,gx,gy,gz) = sol[1:5,end]
#
# Px = r*(x + (1-x-y)*gx) - 1
# Py = r*(x + (1-x-y)*gy)
# Pz = r*(x + (1-x-y)*gz) - g
# P̄ = x*Px + y*Py + (1-x-y)*Pz
# Pz - P̄
