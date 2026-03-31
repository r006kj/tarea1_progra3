#include <iostream>
#include "Tensor.h"

int main() {

    Tensor A = Tensor::zeros({2, 3});
    A.print();

    Tensor B = Tensor::ones({2, 3});
    B.print();

    Tensor C = Tensor::arange(0, 6);
    C.print();

    Tensor X = Tensor({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor Y = Tensor({2, 3}, {6, 5, 4, 3, 2, 1});

    (X + Y).print();
    (X - Y).print();
    (X * Y).print();

    Tensor V = Tensor::arange(0, 6);
    Tensor V2 = V.view({2, 3});
    V2.print();

    Tensor Z = Tensor({2, 3}, {-1, 2, -3, 4, -5, 6});
    ReLU relu;
    Tensor R = Z.apply(relu);
    R.print();

    return 0;
}
