using DifferentialEquations, Distributions, Plots, SpecialFunctions, Sundials

#Parameters
e₁ = 0.01
e₂ = 0.01
ϵ = (1-e₁)*(1-e₂) + e₁*e₂
e = e₂
r = 3
λ = 0.5
bias = 0
tspan = (0.0,100)

function imgscore!(du,u,p,t)
    g = p[1]*u[1] + p[2]*u[2] + (1-p[1]-p[2])*u[3]
    ĝ = λ*g + (1-λ)*bias #min(0.9*g, 1.0)

    # Reputation dynamics: u[1]-u[3]
    Pc = ϵ*ĝ/(ϵ*ĝ + e*(1 - ĝ))
    Pd = (1 - ϵ)*ĝ/((1 - ϵ)*ĝ + (1 - e)*(1 - ĝ))
    Qc = ϵ*Pc + (1 - ϵ)*Pd
    Qd = (1 - e)*Pd + e*Pc
    gx₊ = (1 - u[1])*Qc
    gx₋ = u[1]*(1 - Qc)
    gy₊ = (1 - u[2])*Qd
    gy₋ = u[2]*(1 - Qd)
    gz₊ = (1 - u[3])*(Qc*g + Qd*(p[1]*(1 - u[1]) + p[2]*(1 - u[2]) + (1-p[1]-p[2])*(1 - u[3])))
    gz₋ = u[3]*((1 - Qc)*g + (1 - Qd)*(p[1]*(1 - u[1]) + p[2]*(1 - u[2]) + (1-p[1]-p[2])*(1 - u[3])))
    du[1] = gx₊ - gx₋
    du[2] = gy₊ - gy₋
    du[3] = gz₊ - gz₋
end

# The differentia-algebraic equation (DAE) we wish to solve
function DAE(out,du,u,p,t)
    g = u[1]*u[3] + u[2]*u[4] + (1-u[1]-u[2])*u[5]
    ĝ = λ*g + (1-λ)*bias #min(0.9*g, 1.0)

    #Imitation dynamics: u[1]-u[3]
    Px = r*(u[1] + (1-u[1]-u[2])*u[3]) - 1
    Py = r*(u[1] + (1-u[1]-u[2])*u[4])
    Pz = r*(u[1] + (1-u[1]-u[2])*u[5]) - g
    P̄ = u[1]*Px + u[2]*Py + (1-u[1]-u[2])*Pz
    out[1] = u[1]*(Px - P̄) - du[1]
    out[2] = u[2]*(Py - P̄) - du[2]

    # Reputation dynamics: u[4]-u[6]
    Pc = ϵ*ĝ/(ϵ*ĝ + e*(1 - ĝ))
    Pd = (1 - ϵ)*ĝ/((1 - ϵ)*ĝ + (1 - e)*(1 - ĝ))
    Qc = ϵ*Pc + (1 - ϵ)*Pd
    Qd = (1 - e)*Pd + e*Pc
    gx₊ = (1 - u[3])*Qc
    gx₋ = u[3]*(1 - Qc)
    gy₊ = (1 - u[4])*Qd
    gy₋ = u[4]*(1 - Qd)
    gz₊ = (1 - u[5])*(Qc*g + Qd*(u[1]*(1 - u[3]) + u[2]*(1 - u[4]) + (1-u[1]-u[2])*(1 - u[5])))
    gz₋ = u[5]*((1 - Qc)*g + (1 - Qd)*(u[1]*(1 - u[3]) + u[2]*(1 - u[4]) + (1-u[1]-u[2])*(1 - u[5])))
    out[3] = gx₊ - gx₋
    out[4] = gy₊ - gy₋
    out[5] = gz₊ - gz₋
end

output = zeros(5151,7)
outcontour = Array{Float64}(undef,101,101)
count = 1
for x = 0:0.01:1
    for y = 0:0.01:1-x
        # Initial conditions for u
        prob = ODEProblem(imgscore!,[0.5,0.5,0.5],tspan,[x,y])
        sol = solve(prob)
        (gx, gy, gz) = sol[:,end]
        u₀ = [x, y, gx, gy, gz]
        # Initial condition for du
        g = x*gx + y*gy + (1-x-y)*gz
        ĝ = λ*g + 1 - λ
        Px = r*(x + (1-x-y)*gx) - 1
        Py = r*(x + (1-x-y)*gy)
        Pz = r*(x + (1-x-y)*gz) - g
        P̄ = x*Px + y*Py + (1-x-y)*Pz
        du₀ = [x*(Px - P̄), y*(Py - P̄), 0.0, 0.0, 0.0]
        # Solve DAE
        differential_vars = [true,true,false,false,false]
        prob = DAEProblem(DAE,du₀,u₀,tspan,differential_vars=differential_vars)
        sol = solve(prob,IDA(),linearsolver=:Dense)
        output[count,:] = hcat([x y],sol[:,end]')
        count += 1
        outcontour[round(Int,x*10)+1,round(Int,y*10)+1] = sol[1,end] .+ (1 .- sol[1,end] .- sol[2,end]).*(sol[1,end].*sol[3,end] + sol[2,end].*sol[4,end] .+ (1 .- sol[1,end] .- sol[2,end]).*sol[5,end])
    end
end

using PyCall, PyPlot
current()

numEq = hcat(output[:,1],1 .- output[:,1] .- output[:,2],output[:,2],output[:,1])
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
tax.scatter(numEq[:,1:3], c = cooperation[:,1], linewidths=0.0001, marker=",")
tax.show()
gcf()
# tax.savefig("DAE_ternary")
# clf()
#
# plt = PyPlot.contourf(0:0.01:1, 0:0.01:1, outcontour; levels = collect(0:0.2:1))
# gcf()
