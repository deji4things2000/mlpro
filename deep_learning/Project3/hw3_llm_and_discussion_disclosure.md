# HW3 Disclosure and Discussion

**Course:** COSC 78/278 - Deep Learning  
**Assignment:** Problem Set #3 - Text Generation  
**Date:** May 8, 2026  
**Student Name:** [Your First Middle Last Name]

---

## LLM Tool Usage Disclosure

I used the following LLM tools during this assignment:

Deepseek
Kimi

**Purpose and Extent of Use:**
- I used the LLM to help understand the mathematical formulations for the BasicRNNCell and LSTMCell forward propagation equations.
- I received assistance with debugging shape mismatches in tensor operations, particularly the matrix multiplication dimensions in the forward pass.
- I consulted the LLM for clarification on how to properly implement the ModuleList for stacking RNN layers.

**Example Prompts Used:**
1. "Can you explain the shape of W, V, and b in the BasicRNNCell given vocab_size and hidden_size?"
2. "How do I split the LSTM pre-activation tensor into four parts (input gate, forget gate, output gate, candidate cell)?"
3. "My matrix multiplication is failing with shapes 32x96 and 66x96 - what's wrong?"
4. "What does the error 'modified by an inplace operation' mean in PyTorch and how do I fix it?"
5. "Can you help me understand what the create_sequence.py code is trying to accomplish?"

**How the LLM Supported My Understanding:**
- Explained the need to use `torch.zeros` with proper device placement
- Provided guidance on avoiding in-place operations that break gradient computation

**Limitations Applied:**
- I did not ask for or receive complete code solutions for the core implementations
- All architectural decisions (hidden sizes, sequence length, learning rate) came directly from the assignment specification
- The LLM was used only for conceptual clarification and debugging assistance, not to generate final code
