//
// Created by Us on 30/03/2026.
//

#ifndef PROGRA3_TAREA1_TENSOR_H
#define PROGRA3_TAREA1_TENSOR_H
#include <iostream>
#include <vector>
#include <random>

class Tensor;
Tensor dot(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor add_bias(const Tensor& a, const Tensor& b);

class TensorTransform {
public:
    virtual Tensor apply(const Tensor& t) const = 0;
    virtual ~TensorTransform() = default;
};

class Tensor {
public:
    Tensor(const std::vector<size_t>& shape, const std::vector<double>& values);

    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);
    static Tensor random(const std::vector<size_t>& shape, double min, double max);
    static Tensor arange(int start, int end);

    Tensor apply(const TensorTransform& transform) const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;

    Tensor view(const std::vector<size_t>& new_shape) const;
    Tensor unsqueeze(size_t axis) const;

    static Tensor concat(const std::vector<Tensor>& tensors, size_t axis);

    const std::vector<size_t>& shape() const { return shape_; }
    size_t ndim()       const { return shape_.size(); }
    size_t total_size() const;
    double*       data()       { return data_; }
    const double* data() const { return data_; }

    void print() const;

    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);
    friend Tensor add_bias(const Tensor& a, const Tensor& b);

private:
    double*             data_;
    std::vector<size_t> shape_;
    bool owns_data_;

    void allocate(size_t n);
    void deallocate();
};

class ReLU : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};

class Sigmoid : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};
#endif //PROGRA3_TAREA1_TENSOR_H