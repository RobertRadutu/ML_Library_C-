#include <iostream>
#include <string>
#include <fstream>
#include <memory>
#include <cassert>
#include <unordered_set>

// computational graph
// automatic differentiation
// weak_ptr have to be used to break refrencing cycles, otherwise in the labmda function we would increment the shared_ptr COUNT of RHS, LHS, OUT.


class Value;

using ValuePtr = std::shared_ptr<Value>;

struct Hash{
    size_t operator()(const ValuePtr) const;
};

class Value : public std::enable_shared_from_this<Value>{
public:
    inline static size_t currentId = 0;
    float data;
    float grad;
    std::string op;
    size_t id;
    std::vector<ValuePtr> prev;
    std::function<void()> backward;

    Value(float data, const std::string& op, size_t id) : data(data), op(op), id(id){};

public:
    static ValuePtr create(float data, const std::string& op){
        return std::shared_ptr<Value>(new Value(data, op, currentId++));
    }

    static ValuePtr add(const std::vector<ValuePtr>& operands){
        float result = 0;
        for(auto it : operands){
            result+= it->data;
        }
        auto out = Value::create(result, "+");
        for(auto it : operands) out->prev.push_back(it); 
        out->backward = [out_weak = std::weak_ptr<Value>(out), operands](){
            for(auto it : operands){
                std::weak_ptr aux = std::weak_ptr<Value>(it);
                aux.lock()->grad += out_weak.lock()->grad;
            }
        };
        return out;
    }
    
    static ValuePtr subtract(const std::vector<ValuePtr>& operands){
        auto it = operands.begin();
        float result = (*it)->data;
        for(it = operands.begin()+1; it != operands.end(); it++){
            result -= (*it)->data;
        }
        auto out = Value::create(result, "-");
        for(auto it : operands) out->prev.push_back(it); 
        out->backward = [out_weak = std::weak_ptr<Value>(out), operands](){
            for(auto it : operands){
                std::weak_ptr aux = std::weak_ptr<Value>(it);
                aux.lock()->grad += out_weak.lock()->grad;
            }
        };
        return out;
    }

    static ValuePtr multiply(const std::vector<ValuePtr>& operands){
        float result = 1;
        for(auto it : operands){
            result *= it->data;
        }
        auto out = Value::create(result, "*");
        for(auto it : operands) out->prev.push_back(it); 
        out->backward = [out_weak = std::weak_ptr<Value>(out), operands, result](){
            for(auto it : operands){
                std::weak_ptr aux = std::weak_ptr<Value>(it);
                aux.lock()->grad += out_weak.lock()->grad * (result / it->data);
            }
        };
        return out;
      }

    void buildTopological(const ValuePtr& curr, std::unordered_set<ValuePtr, Hash>& visited, std::vector<std::shared_ptr<Value>>& topo){
        visited.insert(curr);
        for(auto it : curr->prev)
            buildTopological(it, visited, topo);
        topo.push_back(curr);    
    }

    void backProp(){
        std::vector<std::shared_ptr<Value>> topo;
        std::unordered_set<ValuePtr, Hash> visited;
        buildTopological(shared_from_this(), visited, topo);
        reverse(topo.begin(), topo.end());
        std::cout << "Size: " << topo.size() << std::endl;
        this->grad = 1.0f;
        for(auto it = topo.begin(); it != topo.end(); it++){
           // std::cout << (*it)->data << std::endl;
            if((*it)->backward)
                (*it)->backward();
        }
    }
    
    static ValuePtr ReLU(const ValuePtr& x){
       auto out = Value::create(std::max(0.0f, x->data), "ReLU");
       out->prev.push_back(x);
       out->backward = [x_weak = std::weak_ptr<Value>(x), out_weak = std::weak_ptr<Value>(out)](){
            x_weak.lock()->grad += (out_weak.lock()->data > 0) * out_weak.lock()->grad;
       };
       return out;
    }
    
    ~Value(){
        --currentId;
    }
};



size_t Hash::operator()(const ValuePtr value) const{
    return std::hash<std::string>()(value->op) ^ std::hash<float>()(value->data);
}
    

int main(){
    auto a = Value::create(3.0, "");
    auto b = Value::create(7.0, "");
    auto c = Value::create(10.0, "");
    auto d = Value::add({a, b, c});
    auto e = Value::create(5.0, "");
    auto f = Value::create(1.0, "");
    auto g = Value::multiply({d, e, f});
    auto L = Value::ReLU(g);
    
    L->backProp();
 /* 
    std::cout << "Value of L is : " << L->data << std::endl;

    std::cout << "The gradient of L with respect to a is : " << a->grad << std::endl;
    std::cout << "The gradient of L with respect to b is : " << b->grad << std::endl;
    std::cout << "The gradient of L with respect to c is : " << c->grad << std::endl;
    std::cout << "The gradient of L with respect to d is : " << d->grad << std::endl;
    std::cout << "The gradient of L with respect to L is : " << L->grad << std::endl;
    std::cout << "Number of nodes in the graph: " << Value::currentId << std::endl;
*/
    std::cout << b->grad << std::endl;
}
