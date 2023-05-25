#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>


int main(int args, char** argv){
    int n;
    if (args == 2){
        n = atoi(argv[1]);
    }
    else{
        std::cout << "Usage: ./CountSort <number of elements>" << std::endl;
        return 0;
    }
    std::vector<int> a;
    a.resize(n);
    srand((unsigned) time(NULL));
    for (int i = 0; i<n; i++){
        a[i] = (rand() % (1000 + 1));
    }

    // std::cout << "Unsorted array: ";
    // for (int i = 0; i < n; i++)
    //     std::cout << a[i] << " ";
    // std::cout << std::endl;

    cl::sycl::queue queue(0);

    //setup memory
    cl::sycl::buffer<int, 1> d_a(a.data(), n);
    cl::sycl::buffer<int, 1> tempArray(a.data(), n);

    queue.submit([&](cl::sycl::handler& cgh){
        auto d_a_acc = d_a.get_access<cl::sycl::access::mode::read_write>(cgh);
        auto tempArray_acc = tempArray.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<class countSort>(cl::sycl::range<1>(n), [=](cl::sycl::item<1> i){
            int count = 0;
            for (int j = 0; j < n; j++){
                count+= (d_a_acc[j] < d_a_acc[i]);
                count+= (d_a_acc[j] == d_a_acc[i] && j < i);
            }
            tempArray_acc[count] = d_a_acc[i];
        });
        
    });


    queue.wait_and_throw(); 
    
    auto output = tempArray.get_access<cl::sycl::access::mode::read>();

    for (int i = 0; i < n; i++){
        a[i] = output[i];
    }


    //free memory
    // hipFree(d_a);
    // hipFree(tempArray);

    // std::cout << "Sorted array: ";
    // for (int i = 0; i < n; i++)
    //     std::cout << a[i] << " ";
    // std::cout << std::endl;

    //check if the array is sorted
    if (std::is_sorted(a.begin(), a.end()))
        std::cout << "Array is sorted" << std::endl;
    else
        std::cout << "Array is not sorted" << std::endl;

    return 0;
}