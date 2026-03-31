//
// Created by Us on 30/03/2026.
//

#ifndef PROGRA3_TAREA1_TENSOR_H
#define PROGRA3_TAREA1_TENSOR_H
#include <iostream>
#include <vector>
#include <cmath>

class TensorTransform;

class Tensor {
public:
    Tensor(const std::vector<size_t>& shape, const std::vector<double>& values);

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    ~Tensor();

    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);
    static Tensor arange(int start, int end);

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;

    Tensor view(const std::vector<size_t>& new_shape) const;

    Tensor apply(const TensorTransform& transform) const;

    const std::vector<size_t>& shape() const { return shape_; }
    size_t total_size() const;

    double* data() { return data_; }
    const double* data() const { return data_; }

    void print() const;

private:
    double* data_;
    std::vector<size_t> shape_;
};

class TensorTransform {
public:
    virtual Tensor apply(const Tensor& t) const = 0;
    virtual ~TensorTransform() = default;
};

class ReLU : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};

#endif //PROGRA3_TAREA1_TENSOR_H