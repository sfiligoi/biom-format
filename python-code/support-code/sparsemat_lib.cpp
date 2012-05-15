#include <iostream>
#include "sparsemat_lib.h"

using namespace sparsemat;

// SparseMatFloat
SparseMatFloat::SparseMatFloat() {}

void SparseMatFloat::insert(uint32_t row, uint32_t col, double value) {
    current_key = MAKE_KEY(row,col);
#ifdef _SPARSEMAT_LIB_DEBUG
    std::cout << "row << 30: " << (row << 30) << " row << 31: " << (row << 31);
    std::cout << " row << 32: " << (row << 32) << std::endl; 
    std::cout << "Row: " << row << "\tCol: " << col << std::endl;
    std::cout << "Current key: " << current_key << "\tvalue: " << value << std::endl;
#endif
    hash[current_key] = value;
}

double SparseMatFloat::get(uint32_t row, uint32_t col) {
    current_key = MAKE_KEY(row, col);
#ifdef _SPARSEMAT_LIB_DEBUG
    std::cout << "Row: " << row << "\tCol: " << col << std::endl;
    std::cout << "Current key: " << current_key << std::endl;
#endif
    return hash[current_key];
}

// SparseMatInt
SparseMatInt::SparseMatInt() {}

void SparseMatInt::insert(uint32_t row, uint32_t col, int value) {
    current_key = MAKE_KEY(row,col);
#ifdef _SPARSEMAT_LIB_DEBUG
    std::cout << "Row: " << row << "\tCol: " << col << std::endl;
    std::cout << "Current key: " << current_key << "\tvalue: " << value << std::endl;
#endif
    hash[current_key] = value;
}

int SparseMatInt::get(uint32_t row, uint32_t col) {
    current_key = MAKE_KEY(row, col);
#ifdef _SPARSEMAT_LIB_DEBUG
    std::cout << "Row: " << row << "\tCol: " << col << std::endl;
    std::cout << "Current key: " << current_key << std::endl;
#endif
    return hash[current_key];
}
