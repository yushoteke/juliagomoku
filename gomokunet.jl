using Statistics
using Flux

M=randn(5,5)

function toyNN(x::Array{Bool,3})
    return (1 .+rand(5,5))/2,tanh(randn())
end

w,h,c,b=5,5,2,1
body=Chain(Conv((3,3),2=>32,pad=(1,1),leakyrelu),
            Conv((3,3),32=>64,pad=(1,1),leakyrelu),
            Conv((3,3),64=>64,pad=(1,1),leakyrelu),
            Conv((3,3),64=>64,pad=(1,1),leakyrelu),
            x->reshape(x,:,b),
            Dense(64*w*h,512,leakyrelu))
head1=Dense(512,w*h,sigmoid)
head2=Dense(512,1,tanh)

function NN(x::Array{Float32,4})
    w,h,c,b=size(x)
    temp=body(x)
    return reshape(head1(temp),w,h,1,b),reshape(head2(temp),1,1,1,b)
end
