#include <iostream>
#include <vec_test.hpp>

double randval(double left = 0.0, double right = 1.0){
    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<double> dist(left, right);
    return dist(mt);
}

auto creat_v(int n){
    auto start = chrono::steady_clock::now();
    vec v = 100*randu<vec>(n);
    auto end = chrono::steady_clock::now();
    return end - start;
}

auto scale_v(int n){
    vec v = 100*randu<vec>(n);
    double s = randval();
    auto start = chrono::steady_clock::now();
    v *= s;
    auto end = chrono::steady_clock::now();
    return end - start;
}

auto frob_v(int n){
    vec v = 100*randu<vec>(n);
    auto start = chrono::steady_clock::now();
    double f_norm = norm(v, "fro");
    auto end = chrono::steady_clock::now();
    return end - start;
}

auto add_v(int n){
    vec v1 = 100*randu<vec>(n);
    vec v2 = 100*randu<vec>(n);
    auto start = chrono::steady_clock::now();
    vec v3 = v1 + v2;
    auto end = chrono::steady_clock::now();
    return end - start;
}

auto sub_v(int n){
    vec v1 = 100*randu<vec>(n);
    vec v2 = 100*randu<vec>(n);
    auto start = chrono::steady_clock::now();
    vec v3 = v1 - v2;
    auto end = chrono::steady_clock::now();
    return end - start;
}

auto dot_v(int n){
    vec v1 = 100*randu<vec>(n);
    vec v2 = 100*randu<vec>(n);
    auto start = chrono::steady_clock::now();
    double prod = dot(v1,v2);
    auto end = chrono::steady_clock::now();
    return end - start;
}

void test(int r, int n, auto (*f)(int)){
    double diff = 0.0;
    for(int i=0; i < r; i++){
        diff += chrono::duration <double> ((*f)(n)).count();
    }
    cout << "\tVector size : " << n << "\tRepeats : " << r << endl;
    cout << "\tTest duration : " << diff/r << " s" << endl;
}

int main(int argc, char *argv[]){
    int test_case = strtol(argv[1], nullptr, 0);
    int repeats = strtol(argv[2], nullptr, 0);
    int n_size = strtol(argv[3], nullptr, 0);

    switch (test_case)
    {
        case TEST_CREATE:
            cout << "Test#1 Creation of a vector (uniform distribution [0;100])" << endl;
            test(repeats, n_size, creat_v);
            break;
        case TEST_SCALE:
            cout << "Test#2 Scale vector" << endl;
            test(repeats, n_size, scale_v);
            break;
        case TEST_FROB:
            cout << "Test#3 Frobenius norm of a vector" << endl;
            test(repeats, n_size, frob_v);
            break;
        case TEST_ADD:
            cout << "Test#4 Addition of vectors" << endl;
            test(repeats, n_size, add_v);
            break;
        case TEST_SUB:
            cout << "Test#5 Subtarction of vectors" << endl;
            test(repeats, n_size, sub_v);
            break;
        case TEST_DOT:
            cout << "Test#6 Dot prtoduct of vectors" << endl;
            test(repeats, n_size, dot_v);
            break;
        default:
            cout << "Wrong input!" << endl;
            return -1;

    }
    return 0;
}