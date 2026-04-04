#include "Tensor.h"


void Tensor::allocate(size_t n) {
    data_ = new double[n];
    for (size_t i = 0; i < n; i++)
        data_[i] = 0.0;
}

void Tensor::deallocate() {
    if (owns_data_)
        delete[] data_;
    data_ = nullptr;
}

size_t Tensor::total_size() const {
    if (shape_.empty()) return 0;
    size_t total = 1;
    for (size_t i = 0; i < shape_.size(); i++)
        total *= shape_[i];
    return total;
}


Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<double>& values)
    : data_(nullptr), shape_(shape), owns_data_(true)
{
    if (shape_.size() > 3)
        throw "Tensor: máximo 3 dimensiones.";

    size_t n = total_size();
    if (n != values.size())
        throw "Tensor: tamaño inconsistente.";

    allocate(n);
    for (size_t i = 0; i < n; i++)
        data_[i] = values[i];
}

Tensor::Tensor(const Tensor& other)
    : data_(nullptr), shape_(other.shape_), owns_data_(true)
{
    size_t n = other.total_size();
    allocate(n);
    for (size_t i = 0; i < n; i++)
        data_[i] = other.data_[i];
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_), shape_(other.shape_), owns_data_(other.owns_data_)
{
    other.data_ = nullptr;
    other.shape_ = {};
    other.owns_data_ = false;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;

    if (owns_data_) delete[] data_;

    shape_ = other.shape_;
    owns_data_ = true;

    size_t n = other.total_size();
    data_ = new double[n];
    for (size_t i = 0; i < n; i++)
        data_[i] = other.data_[i];

    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;

    if (owns_data_) delete[] data_;

    data_ = other.data_;
    shape_ = other.shape_;
    owns_data_ = other.owns_data_;

    other.data_ = nullptr;
    other.shape_ = {};
    other.owns_data_ = false;

    return *this;
}

Tensor::~Tensor() {
    if (owns_data_)
        delete[] data_;
}


Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (auto s : shape) n *= s;
    std::vector<double> values(n, 0.0);
    return Tensor(shape, values);
}

Tensor Tensor::ones(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (auto s : shape) n *= s;
    std::vector<double> values(n, 1.0);
    return Tensor(shape, values);
}

Tensor Tensor::random(const std::vector<size_t>& shape, double min, double max) {
    size_t n = 1;
    for (auto s : shape) n *= s;

    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(min, max);

    std::vector<double> values(n);
    for (size_t i = 0; i < n; i++)
        values[i] = dist(rng);

    return Tensor(shape, values);
}

Tensor Tensor::arange(int start, int end) {
    if (end <= start)
        throw "arange inválido.";

    size_t n = end - start;
    std::vector<double> values(n);

    for (size_t i = 0; i < n; i++)
        values[i] = start + i;

    return Tensor({n}, values);
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

Tensor Sigmoid::apply(const Tensor& t) const {
    size_t n = t.total_size();
    std::vector<double> out(n);
    for (size_t i = 0; i < n; i++)
        out[i] = 1.0 / (1.0 + exp(-t.data()[i]));
    return Tensor(t.shape(), out);
}

static bool same_shape(const Tensor& a, const Tensor& b) {
    if (a.ndim() != b.ndim()) return false;
    for (size_t i = 0; i < a.ndim(); i++)
        if (a.shape()[i] != b.shape()[i]) return false;
    return true;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (!same_shape(*this, other))
        throw "dim mismatch";

    size_t n = total_size();
    std::vector<double> out(n);
    for (size_t i = 0; i < n; i++)
        out[i] = data_[i] + other.data_[i];

    return Tensor(shape_, out);
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (!same_shape(*this, other))
        throw "dim mismatch";

    size_t n = total_size();
    std::vector<double> out(n);
    for (size_t i = 0; i < n; i++)
        out[i] = data_[i] - other.data_[i];

    return Tensor(shape_, out);
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (!same_shape(*this, other))
        throw "dim mismatch";

    size_t n = total_size();
    std::vector<double> out(n);
    for (size_t i = 0; i < n; i++)
        out[i] = data_[i] * other.data_[i];

    return Tensor(shape_, out);
}

Tensor Tensor::operator*(double scalar) const {
    size_t n = total_size();
    std::vector<double> out(n);
    for (size_t i = 0; i < n; i++)
        out[i] = data_[i] * scalar;

    return Tensor(shape_, out);
}


Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    size_t total = 1;
    for (auto s : new_shape) total *= s;

    if (total != total_size())
        throw "view inválido";

    Tensor t = *this;
    t.shape_ = new_shape;
    t.owns_data_ = false;

    return t;
}

Tensor Tensor::unsqueeze(size_t axis) const {
    if (ndim() >= 3)
        throw "max dims";
    if (axis > ndim())
        throw "axis inválido";

    std::vector<size_t> new_shape;

    for (size_t i = 0; i < ndim(); i++) {
        if (i == axis) new_shape.push_back(1);
        new_shape.push_back(shape_[i]);
    }

    if (axis == ndim()) new_shape.push_back(1);

    Tensor t = *this;
    t.shape_ = new_shape;
    t.owns_data_ = false;

    return t;
}


Tensor dot(const Tensor& a, const Tensor& b) {
    size_t n = a.total_size();
    double sum = 0;
    for (size_t i = 0; i < n; i++)
        sum += a.data()[i] * b.data()[i];
    return Tensor({1}, {sum});
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    size_t M = a.shape()[0];
    size_t K = a.shape()[1];
    size_t N = b.shape()[1];

    std::vector<double> out(M * N, 0.0);

    for (size_t i = 0; i < M; i++)
        for (size_t k = 0; k < K; k++)
            for (size_t j = 0; j < N; j++)
                out[i*N+j] += a.data()[i*K+k] * b.data()[k*N+j];

    return Tensor({M, N}, out);
}

Tensor add_bias(const Tensor& a, const Tensor& b) {
    size_t N = a.shape()[0];
    size_t C = a.shape()[1];

    std::vector<double> out(N*C);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < C; j++)
            out[i*C+j] = a.data()[i*C+j] + b.data()[j];

    return Tensor({N, C}, out);
}

void Tensor::print() const {
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); i++) {
        std::cout << shape_[i];
        if (i + 1 < shape_.size()) std::cout << ", ";
    }
    std::cout << "])\n";
}
