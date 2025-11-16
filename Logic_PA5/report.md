**Course:** Artificial Intelligence  
**Author:** Adedeji Sunday Adediran  
**Institution:** Dartmouth College  
**Date:** November 2025  

## 1. Description

### 1.1 Algorithm Implementations

#### GSAT Algorithm
The GSAT implementation follows the standard algorithm:
1. **Initialization**: Generate random truth assignment
2. **Termination Check**: Verify if all clauses are satisfied
3. **Random Flip**: With probability `threshold`, flip a random variable
4. **Greedy Flip**: Otherwise, score all variables by counting satisfied clauses after flipping, then flip the best variable

**Key Design Decisions**:
- Used dictionary-based assignment for efficient variable access
- Implemented efficient clause evaluation using early termination
- Maintained list of best variables to handle ties randomly

#### WalkSAT Algorithm  
The WalkSAT implementation improves efficiency by:
1. **Focused Search**: Only considers variables from randomly chosen unsatisfied clauses
2. **Reduced Scoring**: Scores only candidate variables from unsatisfied clauses
3. **Balanced Exploration**: Uses threshold to balance random and greedy moves

**Key Design Decisions**:
- Maintained list of unsatisfied clauses for efficient candidate selection
- Used the same scoring mechanism as GSAT but on reduced variable set
- Implemented efficient clause evaluation with literal-based checking

### 1.2 Problem Modeling

The CNF parser handles DIMACS format with:
- Variable numbering from 1 to n
- Clause representation as lists of literals
- Efficient evaluation using sign-based literal interpretation

## 2. Evaluation

### 2.1 Functionality Testing

Both algorithms successfully solve small CNF instances:
- **GSAT**: Effective for problems with 10-100 variables
- **WalkSAT**: Scales better to larger problems (100+ variables)

### 2.2 Performance Analysis

**GSAT Strengths**:
- Simple implementation
- Good for smaller problems
- Thorough search of variable space

**GSAT Limitations**:
- O(n) scoring per iteration becomes expensive for large n
- Poor scaling to Sudoku-sized problems (729 variables)

**WalkSAT Strengths**:
- O(k) scoring where k << n (k = clause size)
- Much faster per iteration
- Better scaling to large problems

**WalkSAT Limitations**:
- May require more iterations to find solution
- More sensitive to threshold parameter tuning

### 2.3 Partial Successes

While full Sudoku puzzles proved challenging within reasonable time limits:
- Both algorithms correctly solve smaller logical puzzles
- The implementations handle CNF format correctly
- Solution verification works reliably
- Parameter tuning shows expected effects on performance

### 2.4 Challenges and Solutions

**Challenge**: Efficient clause evaluation  
**Solution**: Early termination when any literal satisfies clause

**Challenge**: Handling large variable spaces  
**Solution**: WalkSAT's focused candidate selection

**Challenge**: Random tie-breaking  
**Solution**: Maintain list of best candidates and choose randomly

## Conclusion

The implementations successfully demonstrate the GSAT and WalkSAT algorithms. WalkSAT shows superior performance on larger problems due to its focused search strategy, while GSAT provides a solid baseline approach. Further optimizations could include clause weighting or more sophisticated random restart strategies.