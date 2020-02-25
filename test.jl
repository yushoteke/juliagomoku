include("MonteCarloTreeSearch.jl")
include("gomokunet.jl")
#nearly terminal case
G=gomoku(3,2)
n=MCTS_node(G)

simulation(n,toyNN)
simulation(n,toyNN)

play_move!(G,CartesianIndex(1,1))
play_move!(G,CartesianIndex(3,1))
play_move!(G,CartesianIndex(1,3))
play_move!(G,CartesianIndex(3,3))
play_move!(G,CartesianIndex(3,2))

n=MCTS_node(G)
for i=1:10
    simulation(n,toyNN)
end
print_node(n)
print_childs(n)

#terminal case
G=gomoku(3,2)
play_move!(G,CartesianIndex(1,1))
play_move!(G,CartesianIndex(3,1))
play_move!(G,CartesianIndex(1,2))
n=MCTS_node(G)
for i=1:5
    simulation(n,toyNN)
end


G=gomoku(5,3)player
node=MCTS_node(G)
for i=1:100
    simulation(node,NN)
end
PI=zeros(Float32,5,5)
if G.is_terminal
    return PI
else
    for i in keys(node.childs)
        PI[i]=node.childs[i].N
    end
    PI/=sum(PI)
end

p1=player(19,type="res")
p2=player(19,type="res")|>gpu
function compare_times()
    d1=rand(19,19,2,1)
    d2=cu(d1)

    @time p1(d1)
    @time p2(d2)
end
