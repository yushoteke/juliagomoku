using Statistics
using Flux

M=randn(5,5)

function toyNN(x::Array{Bool,3})
    return (1 .+rand(5,5))/2,tanh(randn())
end

w,h,c,b=5,5,2,1
body1=Chain(Conv((3,3),2=>32,pad=(1,1),leakyrelu),
            Conv((3,3),32=>64,pad=(1,1),leakyrelu),
            Conv((3,3),64=>64,pad=(1,1),leakyrelu),
            Conv((3,3),64=>64,pad=(1,1),leakyrelu),
            x->reshape(x,:,b),
            Dense(64*w*h,512,leakyrelu))
head11=Dense(512,w*h,sigmoid)
head12=Dense(512,1,tanh)

function NN1(x::Array{Float32,4})
    w,h,c,b=size(x)
    temp=body1(x)
    return reshape(head11(temp),w,h,1,b),reshape(head12(temp),1,1,1,b)
end

body2=Chain(Conv((3,3),2=>32,pad=(1,1),leakyrelu),
            Conv((3,3),32=>64,pad=(1,1),leakyrelu),
            Conv((3,3),64=>64,pad=(1,1),leakyrelu),
            Conv((3,3),64=>64,pad=(1,1),leakyrelu),
            x->reshape(x,:,b),
            Dense(64*w*h,512,leakyrelu))
head21=Dense(512,w*h,sigmoid)
head22=Dense(512,1,tanh)

function NN2(x::Array{Float32,4})
    w,h,c,b=size(x)
    temp=body2(x)
    return reshape(head21(temp),w,h,1,b),reshape(head22(temp),1,1,1,b)
end
