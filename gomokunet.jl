using Statistics
using Flux

struct player
    body
    head1
    head2
end

Flux.@treelike player

function player(n;type="conv")
    if type=="conv"
        return player(Chain(Conv((3,3),2=>32,pad=(1,1),leakyrelu),
                        Conv((3,3),32=>64,pad=(1,1),leakyrelu),
                        Conv((3,3),64=>64,pad=(1,1),leakyrelu),
                        Conv((3,3),64=>64,pad=(1,1),leakyrelu),
                        x->reshape(x,:,size(x,4)),
                        Dense(64*n*n,512,leakyrelu)),Dense(512,n*n,sigmoid),Dense(512,1,tanh))
    elseif type=="res"
        return player(Chain(Conv((3,3),2=>128,pad=(1,1),gelu),
                            ResidualBlock(128),
                            ResidualBlock(128),
                            ResidualBlock(128),
                            ResidualBlock(128),
                            ResidualBlock(128),
                            Conv((1,1),128=>4),
                            BatchNorm(4),
                            x->gelu.(x),
                            x->reshape(x,:,size(x,4))),
                            Dense(4*n*n,n*n,sigmoid),
                            Chain(Dense(4*n*n,n*n,gelu),Dense(n*n,1,tanh)))
    end
end

function (p::player)(x)
    w,h,c,b=size(x)
    tmp=p.body(x)
    return reshape(p.head1(tmp),w,h,1,b),reshape(p.head2(tmp),1,1,1,b)
end

struct ResidualBlock
    conv1
    norm1
    conv2
    norm2
end

Flux.@treelike ResidualBlock

ResidualBlock(in)=ResidualBlock(Conv((3,3),in=>in,pad=(1,1)),
                                BatchNorm(in),
                                Conv((3,3),in=>in,pad=(1,1)),
                                BatchNorm(in))

function (c::ResidualBlock)(x)
    return gelu.(x.+(gelu.(x|>c.conv1|>c.norm1)|>c.conv2|>c.norm2))
end
