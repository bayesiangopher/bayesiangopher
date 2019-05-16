#include <iostream>
#include <mat_test.hpp>

double randval(double left = 0.0, double right = 1.0){
    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<double> dist(left, right);
    return dist(mt);
}

auto creat_m(int n){
    auto start = chrono::steady_clock::now();
    mat m = 100*randu<mat>(n, n);
    auto end = chrono::steady_clock::now();
    return end - start;
}

auto scale_m(int n){
    mat m = 100*randu<mat>(n, n);
    double s = randval();
    auto start = chrono::steady_clock::now();
    m *= s;
    auto end = chrono::steady_clock::now();
    return end - start;
}

auto trans_m(int n){
    mat m = 100*randu<mat>(n, n);
    auto start = chrono::steady_clock::now();
    m = m.t();
    auto end = chrono::steady_clock::now();
    return end - start;
}

auto add_m(int n){
    mat m1 = 100*randu<mat>(n, n);
    mat m2 = 100*randu<mat>(n, n);
    auto start = chrono::steady_clock::now();
    mat m3 = m1 + m2;
    auto end = chrono::steady_clock::now();
    return end - start;
}

auto sub_m(int n){
    mat m1 = 100*randu<mat>(n, n);
    mat m2 = 100*randu<mat>(n, n);
    auto start = chrono::steady_clock::now();
    mat m3 = m1 - m2;
    auto end = chrono::steady_clock::now();
    return end - start;
}

auto dot_m(int n){
    mat m1 = 100*randu<mat>(n, n);
    mat m2 = 100*randu<mat>(n, n);
    auto start = chrono::steady_clock::now();
    mat m3 = m1 * m2;
    auto end = chrono::steady_clock::now();
    return end - start;
}

auto det_m(int n){
    mat m = 100*randu<mat>(n, n);
    auto start = chrono::steady_clock::now();
    double d = det(m);
    auto end = chrono::steady_clock::now();
    return end - start;
}

auto eigen_m(int n){
    mat m = 100*randu<mat>(n, n);
    cx_vec eigval;
    cx_mat eigvec;
    auto start = chrono::steady_clock::now();
    eig_gen(eigval, eigvec, m);
    auto end = chrono::steady_clock::now();
    return end - start;
}

auto svd_m(int n){
    mat m = 100*randu<mat>(n, n);
    mat U, V;
    vec s;
    auto start = chrono::steady_clock::now();
    svd(U,s,V,m);
    auto end = chrono::steady_clock::now();
    return end - start;
}

auto chol_m(int n){
    mat m1 = 100*randu<mat>(n, n);
    m1 = m1.t()*m1;

    auto start = chrono::steady_clock::now();
    mat m2 = chol(m1); 
    auto end = chrono::steady_clock::now();
    return end - start;
}

void test(int r, int n, auto (*f)(int)){
    double diff = 0.0;
    for(int i=0; i < r; i++){
        diff += chrono::duration <double, nano> ((*f)(n)).count();
    }
    cout << "\tMatrix size : " << n << "\tRepeats : " << r << endl;
    cout << fixed << "Test duration : " << diff/r << " ns" << endl;
}

int main(int argc, char *argv[]){
    int test_case = strtol(argv[1], nullptr, 0);
    int repeats = strtol(argv[2], nullptr, 0);
    int n_size = strtol(argv[3], nullptr, 0);

    switch (test_case)
    {
        case TEST_CREATE:
            cout << "Test#1 Creation of a matrix (uniform distribution [0;100])" << endl;
            test(repeats, n_size, creat_m);
            break;
        case TEST_SCALE:
            cout << "Test#2 Scale matrix" << endl;
            test(repeats, n_size, scale_m);
            break;
        case TEST_TRANSPOSE:
            cout << "Test#3 Frobenius norm of a matrices" << endl;
            test(repeats, n_size, trans_m);
            break;
        case TEST_ADD:
            cout << "Test#4 Addition of matrices" << endl;
            test(repeats, n_size, add_m);
            break;
        case TEST_SUB:
            cout << "Test#5 Subtarction of matrices" << endl;
            test(repeats, n_size, sub_m);
            break;
        case TEST_DOT:
            cout << "Test#6 Dot prtoduct of matrices" << endl;
            test(repeats, n_size, dot_m);
            break;
        case TEST_DET:
            cout << "Test#7 Determinant of a matrix" << endl;
            test(repeats, n_size, det_m);
            break;
        case TEST_EIGEN:
            cout << "Test#8 Eigenvalues and eigenvectors of a matrix" << endl;
            test(repeats, n_size, eigen_m);
            break;
        case TEST_SVD:
            cout << "Test#9 SVD decomposition of a matrix" << endl;
            test(repeats, n_size, svd_m);
            break;
        case TEST_CHOL:
            cout << "Test#10 Cholesky decomposition of a matrix" << endl;
            test(repeats, n_size, chol_m);
            break;
        default:
            cout << "Wrong input!" << endl;
            return -1;

    }
    return 0;
}