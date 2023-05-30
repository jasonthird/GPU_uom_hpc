#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#define maxsize 12000
#define iterations 900
#define ROW 450
#define COL 352
#define start 100
#define accuracy 35

#define MAX_THREADS_PER_BLOCK 32

int main() {
    std::vector<float> table1(maxsize * maxsize);
    std::vector<float> table2(maxsize * maxsize);
    float diff = 0.0f;

    std::fill(table1.begin(), table1.end(), 0.0f);
    std::fill(table2.begin(), table2.end(), 0.0f);

    // pointer to GPU memory
    float *d_table1;
    float *d_table2;
    float *d_diff;

    {
        cl::sycl::queue q;
        cl::sycl::buffer<float, 1> d_table1_buf(table1.data(), cl::sycl::range<1>(maxsize * maxsize));
        cl::sycl::buffer<float, 1> d_table2_buf(table2.data(), cl::sycl::range<1>(maxsize * maxsize));
        cl::sycl::buffer<float, 1> d_diff_buf(&diff, cl::sycl::range<1>(1));

        //Start interation
        for (int k = 0; k < iterations; k++) {
            //init data for each iteration
            q.submit([&](cl::sycl::handler& cgh) {
                auto d_table1_acc = d_table1_buf.get_access<cl::sycl::access::mode::write>(cgh);
                auto d_diff_acc = d_diff_buf.get_access<cl::sycl::access::mode::write>(cgh);
                cgh.single_task<class initData>([=]() {
                    d_table1_acc[ROW * maxsize + COL] = start;
                    d_diff_acc[0] = 0;
                });
            });
            //calculate heat
            q.submit([&](cl::sycl::handler& cgh) {
                auto d_table1_acc = d_table1_buf.get_access<cl::sycl::access::mode::read>(cgh);
                auto d_table2_acc = d_table2_buf.get_access<cl::sycl::access::mode::write>(cgh);
                auto d_diff_acc = d_diff_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
                cgh.parallel_for<class calculateHeat>(cl::sycl::nd_range<2>(cl::sycl::range<2>(maxsize, maxsize), cl::sycl::range<2>(MAX_THREADS_PER_BLOCK, MAX_THREADS_PER_BLOCK)), [=](cl::sycl::nd_item<2> item) {
                    int col = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
                    int row = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
                    if (row >= maxsize || col >= maxsize || row == 0 || row == maxsize - 1 || col == 0 || col == maxsize - 1)
                        return;
                    d_table2_acc[row * maxsize + col] = 0.25 * (d_table1_acc[(row - 1) * maxsize + col] + d_table1_acc[(row + 1) * maxsize + col] + d_table1_acc[row * maxsize + col - 1] + d_table1_acc[row * maxsize + col + 1]);
                    float diff_temp = (d_table2_acc[row * maxsize + col] - d_table1_acc[row * maxsize + col]) * (d_table2_acc[row * maxsize + col] - d_table1_acc[row * maxsize + col]);
                    cl::sycl::atomic_ref<float, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::device> diff_atomic(d_diff_acc[0]);
                    diff_atomic.fetch_add(diff_temp);
                });
            });

            //sqrt diff
            q.submit([&](cl::sycl::handler& cgh) {
                auto d_diff_acc = d_diff_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
                cgh.single_task<class sqrtDiff>([=]() {
                    d_diff_acc[0] = sqrt(d_diff_acc[0]);
                });
            });

            //copy diff from GPU to CPU and check if it is less than accuracy
            diff = d_diff_buf.get_access<cl::sycl::access::mode::read>()[0];
            std::cout << "diff = " << diff << std::endl;
            if (diff < accuracy)
                break;
            //swap table1 and table2
            std::swap(d_table1_buf, d_table2_buf);
        }
    }
  



    std::cout << "final diff = " << diff << std::endl;

    return 0;
}
