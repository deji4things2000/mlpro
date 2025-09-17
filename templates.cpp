
#include <cstdio>
#include <string>

// Define template function that works wth anytype T
template <typename T>
T max_val(T v1, T v2) {
    if (v1 > v2) {
        return v1;
    } else {
        return v2;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s <type> <value1> <value2>\n", argv[0]);
        return 1;
    }

    std::string type = argv[1];
    std::string val1 = argv[2];
    std::string val2 = argv[3];

    if (type == "int") {
        int a = std::stoi(val1);
        int b = std::stoi(val2);
        // TODO: Compute and print the integer case!
        int res = max_val(a,b);
        printf("Max of %d and %d is %d\n", a, b, res);
    }
    else if (type == "float") {
        float a = std::stof(val1);
        float b = std::stof(val2);
        // TODO: Compute and print the float case!
        float res = max_val(a,b);
        printf("Max of %.1f and %.1f is %.1f\n", a, b, res);
    }
    else {
        printf("Unsupported type: %s\n", type.c_str());
        return 1;
    }

    return 0;
}
