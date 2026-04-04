#include "Tensor.h"
#include <iostream>
using namespace std;
void section(const char* title) {
    cout << "  " << title << "\n";
}

int main() {


    section("1. Creacion de tensores");

    Tensor A = Tensor:: zeros({2, 3});
    cout << "zeros({2,3}):   "; A.print();

    Tensor B = Tensor::ones({3, 3});
    cout << "ones({3,3}):   "; B.print();

    Tensor C = Tensor::random({2, 2}, 0.0, 1.0);
    cout << "random({2,2}, 0, 1):   "; C.print();

    Tensor D = Tensor::arange(0, 6);
    cout << "arange(0, 6):    "; D.print();

    section("2. Operadores aritmeticos");

    Tensor X = Tensor({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor Y = Tensor({2, 3}, {6, 5, 4, 3, 2, 1});

    cout << "X:       "; X.print();
    cout << "Y:       "; Y.print();
    cout << "X + Y:   "; (X + Y).print();
    cout << "X - Y:   "; (X - Y).print();
    cout << "X * Y:   "; (X * Y).print();
    cout << "X * 2.0: "; (X * 2.0).print();

    section("3. view y unsqueeze");

    Tensor E = Tensor::arange(0, 12);
    cout << "arange(0,12):             "; E.print();
    cout << "view({3,4}):              "; E.view({3, 4}).print();
    cout << "view({2,2,3}):            "; E.view({2, 2, 3}).print();

    Tensor F = Tensor::arange(0, 3);
    cout << "arange(0,3):              "; F.print();
    cout << "unsqueeze(0) -> {1,3}:    "; F.unsqueeze(0).print();
    cout << "unsqueeze(1) -> {3,1}:    "; F.unsqueeze(1).print();

    section("4. ReLU y Sigmoid (polimorfismo)");

    Tensor G = Tensor::arange(-5, 5).view({2, 5});
    cout << "Input arange(-5,5) view({2,5}): "; G.print();

    ReLU    relu;
    Sigmoid sigmoid;

    Tensor H = G.apply(relu);
    cout << "Despues de ReLU:                "; H.print();

    Tensor I = G.apply(sigmoid);
    cout << "Despues de Sigmoid:             "; I.print();

    section("5. dot y matmul");

    Tensor u = Tensor({3}, {1, 2, 3});
    Tensor v = Tensor({3}, {4, 5, 6});
    cout << "dot([1,2,3], [4,5,6]): "; dot(u, v).print();

    Tensor M1 = Tensor({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor M2 = Tensor({3, 2}, {7, 8, 9, 10, 11, 12});
    cout << "matmul(2x3, 3x2):      "; matmul(M1, M2).print();

    section("6. concat");

    Tensor CA = Tensor::ones({2, 3});
    Tensor CB = Tensor::zeros({2, 3});
    std::cout << "concat axis=0 -> 4x3: "; Tensor::concat({CA, CB}, 0).print();
    std::cout << "concat axis=1 -> 2x6: "; Tensor::concat({CA, CB}, 1).print();

    section("7. Gestion de memoria: copy & move");

    Tensor orig = Tensor({2, 2}, {1, 2, 3, 4});
    cout << "Original:       "; orig.print();

    Tensor copia = orig;
    cout << "Copia profunda: "; copia.print();

    Tensor movido = std::move(orig);
    cout << "Movido:         "; movido.print();

    section("8. Red Neuronal (1000x20x20 -> 1000x10)");

    cout << "Paso 1: entrada 1000x20x20\n";
    Tensor input = Tensor::random({1000, 20, 20}, -1.0, 1.0);
    cout << "  "; input.print();

    cout << "Paso 2: view -> 1000x400\n";
    Tensor flat = input.view({1000, 400});
    cout << "  "; flat.print();

    cout << "Paso 3: matmul con W1 (400x100)\n";
    Tensor W1 = Tensor::random({400, 100}, -0.1, 0.1);
    Tensor z1 = matmul(flat, W1);
    cout << "  "; z1.print();

    cout << "Paso 4: + bias b1 (1x100)\n";
    Tensor b1 = Tensor::zeros({1, 100});
    Tensor a1 = add_bias(z1, b1);
    cout << "  "; a1.print();

    cout << "Paso 5: ReLU\n";
    Tensor r1 = a1.apply(relu);
    cout << "  "; r1.print();

    cout << "Paso 6: matmul con W2 (100x10)\n";
    Tensor W2 = Tensor::random({100, 10}, -0.1, 0.1);
    Tensor z2 = matmul(r1, W2);
    cout << "  "; z2.print();

    cout << "Paso 7: + bias b2 (1x10)\n";
    Tensor b2 = Tensor::zeros({1, 10});
    Tensor a2 = add_bias(z2, b2);
    cout << "  "; a2.print();

    cout << "Paso 8: Sigmoid\n";
    Tensor output = a2.apply(sigmoid);
    cout << "  "; output.print();

    return 0;
}