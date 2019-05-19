#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include <mlpack/methods/pca/pca.hpp>
#include <mlpack/methods/pca/decomposition_policies/exact_svd_method.hpp>
#include <chrono>

using namespace mlpack;
using namespace mlpack::pca;
using namespace mlpack::kmeans;
using namespace mlpack::dbscan;
using namespace mlpack::regression;

template<typename DecompositionPolicy>
//Test cases
#define TEST_LINEAR         1
#define TEST_LOGISTIC       2
#define TEST_KMEANS         3
#define TEST_DBSCAN         4
#define TEST_PCA            5

void linear_fit(int rep);
void logit_fit(int rep);
void kmeans_fit(int rep);
void dbscan_fit(int rep);
void pca_fit(int rep);
