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

    static ValuePtr add(const ValuePtr& lhs, const ValuePtr& rhs){
        auto out = Value::create(lhs->data + rhs->data, "+");
        out->prev = {lhs, rhs};
        out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)](){
            lhs_weak.lock()->grad += out_weak.lock()->grad;
            rhs_weak.lock()->grad += out_weak.lock()->grad;
        };
        return out;
    }
    
    static ValuePtr subtract(const ValuePtr& lhs, const ValuePtr& rhs){
        auto out = Value::create(lhs->data - rhs->data, "-");
        out->prev = {lhs, rhs};
        out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)](){
            lhs_weak.lock()->grad += out_weak.lock()->grad;
            rhs_weak.lock()->grad += out_weak.lock()->grad; 
        };
        return out;
    }

    static ValuePtr multiply(const ValuePtr& lhs, const ValuePtr& rhs){
        auto out = Value::create(lhs->data * rhs->data, "*");
        out->prev = {lhs, rhs};
        out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)](){
            lhs_weak.lock()->grad += out_weak.lock()->grad * rhs_weak.lock()->data;
            rhs_weak.lock()->grad += out_weak.lock()->grad * lhs_weak.lock()->data;
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
            std::cout << (*it)->data << std::endl;
            if((*it)->backward)
                (*it)->backward();
        }
    }

    ~Value(){
        --currentId;
    }
};

size_t Hash::operator()(const ValuePtr value) const{
    return std::hash<std::string>()(value->op) ^ std::hash<float>()(value->data);
}
    

int main(){
    auto a = Value::create(1.0, "");
    auto b = Value::create(2.0, "");
    auto c = Value::add(a, b);
    auto d = Value::create(4.0, "");
    auto L = Value::multiply(c, d);
    L->backProp();
    std::cout << "The gradient of L with respect to a is : " << a->grad << std::endl;
    std::cout << "The gradient of L with respect to b is : " << b->grad << std::endl;
    std::cout << "The gradient of L with respect to c is : " << c->grad << std::endl;
    std::cout << "The gradient of L with respect to d is : " << d->grad << std::endl;
    std::cout << "The gradient of L with respect to L is : " << L->grad << std::endl;
}