#include <iostream>
#include <mlpack_tests.hpp>

using namespace std;

void linear_fit(int rep){
    arma::mat x_train;
    arma::vec y_train;

    data::Load("/home/opqi/go/src/github.com/bayesiangopher/bayesiangopher/datasets/the_WWT_weather_dataset_xtrain.csv", x_train, true);
    y_train.load("/home/opqi/go/src/github.com/bayesiangopher/bayesiangopher/datasets/the_WWT_weather_dataset_ytrain.csv");

    LinearRegression lr;
    double diff = 0.0;
    for (int i = 0; i < rep; i++){
        auto start = chrono::steady_clock::now();
        lr.Train(x_train, y_train);
        auto end = chrono::steady_clock::now();
        diff += chrono::duration <double, nano> (end - start).count();
    }
    cout << fixed << "Test duration : " << diff/(float)rep << " ns" << endl;
}

void logit_fit(int rep){
    arma::mat x_train;
    arma::mat y_train;

    data::Load("/home/opqi/go/src/github.com/bayesiangopher/bayesiangopher/datasets/the_breast_canser_dataset_xtrain.csv", x_train, true);
    data::Load("/home/opqi/go/src/github.com/bayesiangopher/bayesiangopher/datasets/the_breast_canser_dataset_ytrain.csv", y_train, true);

    arma::Row<size_t> responses(y_train.n_cols);

    for (int i = 0; i < y_train.n_cols; i++){
        responses[i] = (size_t)y_train.row(0)[i];
    }

    LogisticRegression<> logit(x_train.n_rows);
    double diff = 0.0;
    for (int i = 0; i < rep; i++){
        auto start = chrono::steady_clock::now();
        logit.Train(x_train, responses);
        auto end = chrono::steady_clock::now();
        diff += chrono::duration <double, nano> (end - start).count();
    }
    cout << fixed << "Test duration : " << diff/(float)rep << " ns" << endl;
}

void kmeans_fit(int rep){
    arma::mat x_train;
    data::Load("/home/opqi/go/src/github.com/bayesiangopher/bayesiangopher/datasets/the_xclara_cluster_train.csv", x_train, true);

    arma::Row<size_t> assignments;
    arma::size_t clusters = 3;

    KMeans<> k;
    double diff = 0.0;
    for (int i = 0; i < rep; i++){
        auto start = chrono::steady_clock::now();
        k.Cluster(x_train, clusters, assignments);
        auto end = chrono::steady_clock::now();
        diff += chrono::duration <double, nano> (end - start).count();
    }
    cout << fixed << "Test duration : " << diff/(float)rep << " ns" << endl;
}

void dbscan_fit(int rep){
    arma::mat x_train;
    data::Load("/home/opqi/go/src/github.com/bayesiangopher/bayesiangopher/datasets/the_dbscan_handmade_dataset.csv", x_train, true);

    arma::Row<size_t> assignments;

    DBSCAN<> db(0.3, 7);
    double diff = 0.0;
    for (int i = 0; i < rep; i++){
        auto start = chrono::steady_clock::now();
        db.Cluster(x_train, assignments);
        auto end = chrono::steady_clock::now();
        diff += chrono::duration <double, nano> (end - start).count();
    }
    cout << fixed << "Test duration : " << diff/(float)rep << " ns" << endl;
}

void pca_fit(int rep){
    arma::mat x_train;
    data::Load("/home/opqi/go/src/github.com/bayesiangopher/bayesiangopher/datasets/the_boston_housing_dataset_train.csv", x_train, true);

    PCAType<> pca;
    double diff = 0.0;
    for (int i = 0; i < rep; i++){
        auto start = chrono::steady_clock::now();
        pca.Apply(x_train, 2);
        auto end = chrono::steady_clock::now();
        diff += chrono::duration <double, nano> (end - start).count();
    }
    cout << fixed << "Test duration : " << diff/(float)rep << " ns" << endl;
}


int main(int argc, char* argv[]){
    int test_case = strtol(argv[1], nullptr, 0);
    int repeats = strtol(argv[2], nullptr, 0);

    switch (test_case)
        {
            case TEST_LINEAR:
                cout << "Test#1 Linear Regression" << endl;
                linear_fit(repeats);
                break;
            case TEST_LOGISTIC:
                cout << "Test#2 Logistic Regression" << endl;
                logit_fit(repeats);
                break;
            case TEST_KMEANS:
                cout << "Test#3 KMeans" << endl;
                kmeans_fit(repeats);
                break;
            case TEST_DBSCAN:
                cout << "Test#4 DBSCAN" << endl;
                dbscan_fit(repeats);
                break;
            case TEST_PCA:
                cout << "Test#5 PCA" << endl;
                pca_fit(repeats);
                break;
            default:
                cout << "Wrong input!" << endl;
                return -1;

        }
    return 0;
}