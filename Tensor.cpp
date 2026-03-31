//
// Created by Us on 30/03/2026.
//
#include "Tensor.h"


size_t Tensor::total_size() const {
    size_t total = 1;
    for (size_t i = 0; i < shape_.size(); i++)
        total *= shape_[i];
    return total;
}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<double>& values)
    : shape_(shape)
{
    size_t n = total_size();
    data_ = new double[n];

    for (size_t i = 0; i < n; i++)
        data_[i] = values[i];
}

Tensor::Tensor(const Tensor& other) : shape_(other.shape_) {
    size_t n = other.total_size();
    data_ = new double[n];
    for (size_t i = 0; i < n; i++)
        data_[i] = other.data_[i];
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;

    delete[] data_;
    shape_ = other.shape_;

    size_t n = other.total_size();
    data_ = new double[n];

    for (size_t i = 0; i < n; i++)
        data_[i] = other.data_[i];

    return *this;
}

Tensor::~Tensor() {
    delete[] data_;
}


Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (auto s : shape) n *= s;
    std::vector<double> vals(n, 0.0);
    return Tensor(shape, vals);
}

Tensor Tensor::ones(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (auto s : shape) n *= s;
    std::vector<double> vals(n, 1.0);
    return Tensor(shape, vals);
}

Tensor Tensor::arange(int start, int end) {
    size_t n = end - start;
    std::vector<double> vals(n);
    for (size_t i = 0; i < n; i++)
        vals[i] = start + i;
    return Tensor({n}, vals);
}

// ===== operators =====

Tensor Tensor::operator+(const Tensor& other) const {
    size_t n = total_size();
    std::vector<double> out(n);

    for (size_t i = 0; i < n; i++)
        out[i] = data_[i] + other.data_[i];

    return Tensor(shape_, out);
}

Tensor Tensor::operator-(const Tensor& other) const {
    size_t n = total_size();
    std::vector<double> out(n);

    for (size_t i = 0; i < n; i++)
        out[i] = data_[i] - other.data_[i];

    return Tensor(shape_, out);
}

Tensor Tensor::operator*(const Tensor& other) const {
    size_t n = total_size();
    std::vector<double> out(n);

    for (size_t i = 0; i < n; i++)
        out[i] = data_[i] * other.data_[i];

    return Tensor(shape_, out);
}



Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    size_t n = total_size();
    std::vector<double> vals(n);

    for (size_t i = 0; i < n; i++)
        vals[i] = data_[i];

    return Tensor(new_shape, vals);
}


Tensor Tensor::apply(const TensorTransform& transform) const {
    return transform.apply(*this);
}

Tensor ReLU::apply(const Tensor& t) const {
    size_t n = t.total_size();
    std::vector<double> out(n);

    for (size_t i = 0; i < n; i++)
        out[i] = t.data()[i] > 0 ? t.data()[i] : 0;

    return Tensor(t.shape(), out);
}

void Tensor::print() const {
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); i++) {
        std::cout << shape_[i];
        if (i + 1 < shape_.size()) std::cout << ",";
    }
    std::cout << "])\n";
}