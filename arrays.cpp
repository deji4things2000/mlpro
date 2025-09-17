
#include <cstdio>       // for printf
#include <string>       // for std::stoi

// Another function for printing
void print_array(int arr[], int size) {
    printf("Array elements: ");
    for (int i = 0; i < size; i++) {
        printf("%d, ", arr[i]);  //print each element followed by comma
    }
    printf("\n"); //move the cursor to a new line
}

int main(int argc, char *argv[]) {

    // Check if the correct number of arguments are provided
    if (argc < 2) {
        printf("Usage: %s <number>\n", argv[0]); //similar function as ps 1
        return 1;
    }

    // Convert the first argument to an integer
    int input_val = std::stoi(argv[1]);

    int arr[10];
    int sum = 0;

    // Fill the array with the specified values and print the array
    arr[0] = input_val;
    for (int i = 1; i <= 9; i++) {
        arr[i] = arr[i-1] + 1;
    }

    // Calculate the sum of array elements and print the sum
    for (int i = 0; i<=9; i++) {
        sum+=arr[i];
    }
    
    //Print array
    print_array(arr, 10);

    //Print sum
    printf("Sum: %d\n", sum);

    return 0;
}
