// MIT License
// 
// Copyright (c) 2020 Yaqing Ding, Nanjing University of Science and Technology
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <Eigen/Dense>
#include <mex.h>

using namespace Eigen;

MatrixXd solver_var(const VectorXd &data)
{
    // Compute coefficients
    const double *d = data.data();
    VectorXd coeffs(76);
    coeffs[0] = d[0] + d[9]*d[1] + d[16]*(d[3] + d[9]*d[4]);
    coeffs[1] = d[10]*(d[0] + d[16]*d[3]) + d[11]*(d[1] + d[16]*d[4]);
    coeffs[2] = d[12]*(d[2] + d[16]*d[5]);
    coeffs[3] = d[13]*(d[0] + d[16]*d[3]) + d[14]*(d[1] + d[16]*d[4]);
    coeffs[4] = d[15]*(d[2] + d[16]*d[5]);
    coeffs[5] = d[17]*(d[0] + d[9]*d[1]) + d[18]*(d[3]  + d[9]*d[4]);
    coeffs[6] = d[19]*(d[6] + d[9]*d[7]);
    coeffs[7] = d[10]*(d[17]*d[0] + d[18]*d[3]) + d[11]*(d[17]*d[1] + d[18]*d[4]);
    coeffs[8] = d[12]*(d[17]*d[2] + d[18]*d[5]);
    coeffs[9] = d[19]*(d[10]*d[6] + d[11]*d[7]);
    coeffs[10] = d[12]*d[19]*d[8];
    coeffs[11] = d[13]*(d[17]*d[0]+ d[18]*d[3]) + d[14]*(d[17]*d[1]  + d[18]*d[4]);
    coeffs[12] = d[15]*(d[17]*d[2] + d[18]*d[5]);
    coeffs[13] = d[19]*(d[13]*d[6] + d[14]*d[7]);
    coeffs[14] = d[15]*d[19]*d[8];
    coeffs[15] = d[20]*(d[0] + d[9]*d[1]) + d[21]*(d[3]  + d[9]*d[4]);
    coeffs[16] = d[22]*d[6] + d[9]*d[22]*d[7];
    coeffs[17] = d[10]*(d[20]*d[0]+ d[21]*d[3]) + d[11]*(d[20]*d[1]  + d[21]*d[4]);
    coeffs[18] = d[12]*(d[20]*d[2] + d[21]*d[5]);
    coeffs[19] = d[22]*(d[10]*d[6] + d[11]*d[7]);
    coeffs[20] = d[12]*d[22]*d[8];
    coeffs[21] = d[13]*(d[20]*d[0] + d[21]*d[3]) + d[14]*(d[20]*d[1]  + d[21]*d[4]);
    coeffs[22] = d[15]*(d[20]*d[2] + d[21]*d[5]);
    coeffs[23] = d[22]*(d[13]*d[6] + d[14]*d[7]);
    coeffs[24] = d[15]*d[22]*d[8];
    
    // coefficients of the first polynomial
    coeffs[25] = coeffs[1]*coeffs[11] - coeffs[3]*coeffs[7] + coeffs[5]*coeffs[17] - coeffs[7]*coeffs[15];
    coeffs[26] = coeffs[3] + coeffs[15];
    coeffs[27] = coeffs[1]*coeffs[12] - coeffs[3]*coeffs[8] - coeffs[7]*coeffs[4] + coeffs[11]*coeffs[2] + coeffs[5]*coeffs[18] - coeffs[15]*coeffs[8];
    coeffs[28] = coeffs[4];
    coeffs[29] = coeffs[17]*coeffs[6] - coeffs[7]*coeffs[16] + coeffs[1]*coeffs[13] - coeffs[3]*coeffs[9] + coeffs[5]*coeffs[19] - coeffs[15]*coeffs[9];
    coeffs[30] = coeffs[16];
    coeffs[31] = coeffs[6]*coeffs[18] - coeffs[8]*coeffs[16] + coeffs[1]*coeffs[14] - coeffs[3]*coeffs[10] + coeffs[2]*coeffs[13] - coeffs[4]*coeffs[9] + coeffs[5]*coeffs[20] - coeffs[15]*coeffs[10];
    
    // coefficients of the second polynomial
    coeffs[32] = coeffs[1]*coeffs[5] - coeffs[0]*coeffs[7] + coeffs[7]*coeffs[21] - coeffs[11]*coeffs[17];
    coeffs[33] = coeffs[0] - coeffs[21];
    coeffs[34] = coeffs[5]*coeffs[2] - coeffs[0]*coeffs[8] + coeffs[7]*coeffs[22] - coeffs[11]*coeffs[18] - coeffs[17]*coeffs[12] + coeffs[21]*coeffs[8];
    coeffs[35] = -coeffs[22];
    coeffs[36] = coeffs[1]*coeffs[6] - coeffs[0]*coeffs[9] + coeffs[7]*coeffs[23] - coeffs[11]*coeffs[19] - coeffs[17]*coeffs[13] + coeffs[21]*coeffs[9];
    coeffs[37] = -coeffs[23];
    coeffs[38] = coeffs[2]*coeffs[6] - coeffs[0]*coeffs[10] + coeffs[7]*coeffs[24] - coeffs[11]*coeffs[20] - coeffs[17]*coeffs[14] + coeffs[21]*coeffs[10] + coeffs[8]*coeffs[23] - coeffs[12]*coeffs[19] - coeffs[18]*coeffs[13] + coeffs[22]*coeffs[9];
    coeffs[39] = -coeffs[24];
    
    // coefficients of the third polynomial
    coeffs[40] = - coeffs[0]*coeffs[0]*coeffs[11]*coeffs[11] + 2*coeffs[0]*coeffs[3]*coeffs[5]*coeffs[11] - coeffs[3]*coeffs[3]*coeffs[5]*coeffs[5] - coeffs[5]*coeffs[5]*coeffs[21]*coeffs[21] + 2*coeffs[5]*coeffs[11]*coeffs[15]*coeffs[21] - coeffs[11]*coeffs[11]*coeffs[15]*coeffs[15];
    coeffs[41] = coeffs[5]*coeffs[5] + coeffs[11]*coeffs[11];
    coeffs[42] = 2*(coeffs[0]*coeffs[3]*coeffs[5]*coeffs[12] - coeffs[0]*coeffs[0]*coeffs[11]*coeffs[12] - coeffs[11]*coeffs[15]*coeffs[15]*coeffs[12] - coeffs[5]*coeffs[5]*coeffs[21]*coeffs[22] - coeffs[3]*coeffs[5]*coeffs[5]*coeffs[4] + coeffs[0]*coeffs[5]*coeffs[11]*coeffs[4] + coeffs[5]*coeffs[11]*coeffs[15]*coeffs[22] + coeffs[5]*coeffs[15]*coeffs[21]*coeffs[12]);
    coeffs[43] = 2*coeffs[11]*coeffs[12];
    coeffs[44] = - coeffs[0]*coeffs[0]*coeffs[12]*coeffs[12] + 2*coeffs[0]*coeffs[5]*coeffs[4]*coeffs[12] - coeffs[5]*coeffs[5]*coeffs[4]*coeffs[4] - coeffs[5]*coeffs[5]*coeffs[22]*coeffs[22] + 2*coeffs[5]*coeffs[15]*coeffs[12]*coeffs[22] - coeffs[15]*coeffs[15]*coeffs[12]*coeffs[12];
    coeffs[45] = coeffs[12]*coeffs[12];
    coeffs[46] = 2*(coeffs[13]*coeffs[0]*coeffs[3]*coeffs[5] - coeffs[13]*coeffs[0]*coeffs[0]*coeffs[11] + coeffs[6]*coeffs[0]*coeffs[3]*coeffs[11] - coeffs[6]*coeffs[3]*coeffs[3]*coeffs[5] - coeffs[23]*coeffs[5]*coeffs[5]*coeffs[21] + coeffs[23]*coeffs[5]*coeffs[11]*coeffs[15] + coeffs[16]*coeffs[5]*coeffs[11]*coeffs[21] + coeffs[13]*coeffs[5]*coeffs[15]*coeffs[21] - coeffs[6]*coeffs[5]*coeffs[21]*coeffs[21] - coeffs[16]*coeffs[11]*coeffs[11]*coeffs[15] - coeffs[13]*coeffs[11]*coeffs[15]*coeffs[15] + coeffs[6]*coeffs[11]*coeffs[15]*coeffs[21]);
    coeffs[47] = 2*(coeffs[5]*coeffs[6] + coeffs[11]*coeffs[13]);
    coeffs[48] = 2*(coeffs[0]*coeffs[3]*coeffs[6]*coeffs[12] - coeffs[0]*coeffs[0]*coeffs[12]*coeffs[13] - coeffs[11]*coeffs[15]*coeffs[15]*coeffs[14] - coeffs[15]*coeffs[15]*coeffs[12]*coeffs[13] - coeffs[5]*coeffs[5]*coeffs[21]*coeffs[24] - coeffs[5]*coeffs[5]*coeffs[22]*coeffs[23] - coeffs[0]*coeffs[0]*coeffs[11]*coeffs[14] + coeffs[0]*coeffs[11]*coeffs[4]*coeffs[6] - 2*coeffs[3]*coeffs[5]*coeffs[4]*coeffs[6] + coeffs[5]*coeffs[11]*coeffs[16]*coeffs[22] - 2*coeffs[5]*coeffs[21]*coeffs[6]*coeffs[22] + coeffs[5]*coeffs[21]*coeffs[12]*coeffs[16] + coeffs[11]*coeffs[15]*coeffs[6]*coeffs[22] - 2*coeffs[11]*coeffs[15]*coeffs[12]*coeffs[16] + coeffs[15]*coeffs[21]*coeffs[6]*coeffs[12] + coeffs[0]*coeffs[3]*coeffs[5]*coeffs[14] + coeffs[0]*coeffs[5]*coeffs[4]*coeffs[13] + coeffs[5]*coeffs[11]*coeffs[15]*coeffs[24] + coeffs[5]*coeffs[15]*coeffs[21]*coeffs[14] + coeffs[5]*coeffs[15]*coeffs[12]*coeffs[23] + coeffs[5]*coeffs[15]*coeffs[22]*coeffs[13]);
    coeffs[49] = 2*(coeffs[11]*coeffs[14] + coeffs[12]*coeffs[13]);
    coeffs[50] = 2*(coeffs[14]*coeffs[0]*coeffs[5]*coeffs[4] - coeffs[14]*coeffs[0]*coeffs[0]*coeffs[12] + coeffs[6]*coeffs[0]*coeffs[4]*coeffs[12] - coeffs[24]*coeffs[5]*coeffs[5]*coeffs[22] + coeffs[24]*coeffs[5]*coeffs[15]*coeffs[12] + coeffs[14]*coeffs[5]*coeffs[15]*coeffs[22] - coeffs[6]*coeffs[5]*coeffs[4]*coeffs[4] + coeffs[16]*coeffs[5]*coeffs[12]*coeffs[22] - coeffs[6]*coeffs[5]*coeffs[22]*coeffs[22] - coeffs[14]*coeffs[15]*coeffs[15]*coeffs[12] - coeffs[16]*coeffs[15]*coeffs[12]*coeffs[12] + coeffs[6]*coeffs[15]*coeffs[12]*coeffs[22]);
    coeffs[51] = 2*coeffs[12]*coeffs[14];
    coeffs[52] = - coeffs[0]*coeffs[0]*coeffs[13]*coeffs[13] + 2*coeffs[0]*coeffs[3]*coeffs[6]*coeffs[13] - coeffs[3]*coeffs[3]*coeffs[6]*coeffs[6] - coeffs[5]*coeffs[5]*coeffs[23]*coeffs[23] + 2*coeffs[5]*coeffs[11]*coeffs[16]*coeffs[23] + 2*coeffs[5]*coeffs[15]*coeffs[13]*coeffs[23] - 4*coeffs[5]*coeffs[21]*coeffs[6]*coeffs[23] + 2*coeffs[5]*coeffs[21]*coeffs[16]*coeffs[13] - coeffs[11]*coeffs[11]*coeffs[16]*coeffs[16] + 2*coeffs[11]*coeffs[15]*coeffs[6]*coeffs[23] - 4*coeffs[11]*coeffs[15]*coeffs[16]*coeffs[13] + 2*coeffs[11]*coeffs[21]*coeffs[6]*coeffs[16] - coeffs[15]*coeffs[15]*coeffs[13]*coeffs[13] + 2*coeffs[15]*coeffs[21]*coeffs[6]*coeffs[13] - coeffs[21]*coeffs[21]*coeffs[6]*coeffs[6];
    coeffs[53] = coeffs[6]*coeffs[6] + coeffs[13]*coeffs[13];
    coeffs[54] = 2*(coeffs[11]*coeffs[6]*coeffs[16]*coeffs[22] - coeffs[11]*coeffs[12]*coeffs[16]*coeffs[16] - coeffs[21]*coeffs[6]*coeffs[6]*coeffs[22] - coeffs[0]*coeffs[0]*coeffs[13]*coeffs[14] - coeffs[15]*coeffs[15]*coeffs[13]*coeffs[14] - coeffs[5]*coeffs[5]*coeffs[23]*coeffs[24] - coeffs[3]*coeffs[4]*coeffs[6]*coeffs[6] + coeffs[21]*coeffs[6]*coeffs[12]*coeffs[16] + coeffs[0]*coeffs[3]*coeffs[6]*coeffs[14] + coeffs[0]*coeffs[4]*coeffs[6]*coeffs[13] + coeffs[5]*coeffs[11]*coeffs[16]*coeffs[24] - 2*coeffs[5]*coeffs[21]*coeffs[6]*coeffs[24] + coeffs[5]*coeffs[21]*coeffs[16]*coeffs[14] - 2*coeffs[5]*coeffs[6]*coeffs[22]*coeffs[23] + coeffs[5]*coeffs[12]*coeffs[16]*coeffs[23] + coeffs[5]*coeffs[16]*coeffs[22]*coeffs[13] + coeffs[11]*coeffs[15]*coeffs[6]*coeffs[24] - 2*coeffs[11]*coeffs[15]*coeffs[16]*coeffs[14] + coeffs[15]*coeffs[21]*coeffs[6]*coeffs[14] + coeffs[15]*coeffs[6]*coeffs[12]*coeffs[23] + coeffs[15]*coeffs[6]*coeffs[22]*coeffs[13] - 2*coeffs[15]*coeffs[12]*coeffs[16]*coeffs[13] + coeffs[5]*coeffs[15]*coeffs[13]*coeffs[24] + coeffs[5]*coeffs[15]*coeffs[23]*coeffs[14]);
    coeffs[55] = 2*coeffs[13]*coeffs[14];
    coeffs[56] = - coeffs[0]*coeffs[0]*coeffs[14]*coeffs[14] + 2*coeffs[0]*coeffs[4]*coeffs[6]*coeffs[14] - coeffs[5]*coeffs[5]*coeffs[24]*coeffs[24] + 2*coeffs[5]*coeffs[15]*coeffs[14]*coeffs[24] - 4*coeffs[5]*coeffs[6]*coeffs[22]*coeffs[24] + 2*coeffs[5]*coeffs[12]*coeffs[16]*coeffs[24] + 2*coeffs[5]*coeffs[16]*coeffs[22]*coeffs[14] - coeffs[15]*coeffs[15]*coeffs[14]*coeffs[14] + 2*coeffs[15]*coeffs[6]*coeffs[12]*coeffs[24] + 2*coeffs[15]*coeffs[6]*coeffs[22]*coeffs[14] - 4*coeffs[15]*coeffs[12]*coeffs[16]*coeffs[14] - coeffs[4]*coeffs[4]*coeffs[6]*coeffs[6] - coeffs[6]*coeffs[6]*coeffs[22]*coeffs[22] + 2*coeffs[6]*coeffs[12]*coeffs[16]*coeffs[22] - coeffs[12]*coeffs[12]*coeffs[16]*coeffs[16];
    coeffs[57] = coeffs[14]*coeffs[14];
    
    // coefficients of the fourth polynomial
    coeffs[58] = - coeffs[0]*coeffs[0]*coeffs[17]*coeffs[17] + 2*coeffs[0]*coeffs[1]*coeffs[15]*coeffs[17] - coeffs[1]*coeffs[1]*coeffs[15]*coeffs[15] - coeffs[1]*coeffs[1]*coeffs[21]*coeffs[21] + 2*coeffs[1]*coeffs[3]*coeffs[17]*coeffs[21] - coeffs[3]*coeffs[3]*coeffs[17]*coeffs[17];
    coeffs[59] = coeffs[1]*coeffs[1] + coeffs[17]*coeffs[17];
    coeffs[60] = - 2*coeffs[18]*coeffs[0]*coeffs[0]*coeffs[17] + 2*coeffs[18]*coeffs[0]*coeffs[1]*coeffs[15] + 2*coeffs[2]*coeffs[0]*coeffs[15]*coeffs[17] - 2*coeffs[22]*coeffs[1]*coeffs[1]*coeffs[21] + 2*coeffs[22]*coeffs[1]*coeffs[3]*coeffs[17] + 2*coeffs[18]*coeffs[1]*coeffs[3]*coeffs[21] - 2*coeffs[2]*coeffs[1]*coeffs[15]*coeffs[15] + 2*coeffs[4]*coeffs[1]*coeffs[17]*coeffs[21] - 2*coeffs[2]*coeffs[1]*coeffs[21]*coeffs[21] - 2*coeffs[18]*coeffs[3]*coeffs[3]*coeffs[17] - 2*coeffs[4]*coeffs[3]*coeffs[17]*coeffs[17] + 2*coeffs[2]*coeffs[3]*coeffs[17]*coeffs[21];
    coeffs[61] = 2*coeffs[1]*coeffs[2] + 2*coeffs[17]*coeffs[18];
    coeffs[62] = - coeffs[0]*coeffs[0]*coeffs[18]*coeffs[18] + 2*coeffs[0]*coeffs[15]*coeffs[2]*coeffs[18] - coeffs[1]*coeffs[1]*coeffs[22]*coeffs[22] + 2*coeffs[1]*coeffs[3]*coeffs[18]*coeffs[22] + 2*coeffs[1]*coeffs[17]*coeffs[4]*coeffs[22] - 4*coeffs[1]*coeffs[21]*coeffs[2]*coeffs[22] + 2*coeffs[1]*coeffs[21]*coeffs[4]*coeffs[18] - coeffs[3]*coeffs[3]*coeffs[18]*coeffs[18] + 2*coeffs[3]*coeffs[17]*coeffs[2]*coeffs[22] - 4*coeffs[3]*coeffs[17]*coeffs[4]*coeffs[18] + 2*coeffs[3]*coeffs[21]*coeffs[2]*coeffs[18] - coeffs[15]*coeffs[15]*coeffs[2]*coeffs[2] - coeffs[17]*coeffs[17]*coeffs[4]*coeffs[4] + 2*coeffs[17]*coeffs[21]*coeffs[2]*coeffs[4] - coeffs[21]*coeffs[21]*coeffs[2]*coeffs[2];
    coeffs[63] = coeffs[2]*coeffs[2] + coeffs[18]*coeffs[18];
    coeffs[64] = 2*coeffs[0]*coeffs[1]*coeffs[17]*coeffs[16] - 2*coeffs[0]*coeffs[0]*coeffs[17]*coeffs[19] - 2*coeffs[3]*coeffs[3]*coeffs[17]*coeffs[19] - 2*coeffs[1]*coeffs[1]*coeffs[21]*coeffs[23] - 2*coeffs[1]*coeffs[1]*coeffs[15]*coeffs[16] + 2*coeffs[0]*coeffs[1]*coeffs[15]*coeffs[19] + 2*coeffs[1]*coeffs[3]*coeffs[17]*coeffs[23] + 2*coeffs[1]*coeffs[3]*coeffs[21]*coeffs[19];
    coeffs[65] = 2*coeffs[17]*coeffs[19];
    coeffs[66] = 2*coeffs[0]*coeffs[1]*coeffs[16]*coeffs[18] - 2*coeffs[0]*coeffs[0]*coeffs[18]*coeffs[19] - 2*coeffs[3]*coeffs[3]*coeffs[17]*coeffs[20] - 2*coeffs[3]*coeffs[3]*coeffs[18]*coeffs[19] - 2*coeffs[1]*coeffs[1]*coeffs[21]*coeffs[24] - 2*coeffs[1]*coeffs[1]*coeffs[22]*coeffs[23] - 2*coeffs[0]*coeffs[0]*coeffs[17]*coeffs[20] + 2*coeffs[0]*coeffs[17]*coeffs[2]*coeffs[16] - 4*coeffs[1]*coeffs[15]*coeffs[2]*coeffs[16] + 2*coeffs[0]*coeffs[1]*coeffs[15]*coeffs[20] + 2*coeffs[0]*coeffs[15]*coeffs[2]*coeffs[19] + 2*coeffs[1]*coeffs[3]*coeffs[17]*coeffs[24] + 2*coeffs[1]*coeffs[3]*coeffs[21]*coeffs[20] + 2*coeffs[1]*coeffs[3]*coeffs[18]*coeffs[23] + 2*coeffs[1]*coeffs[3]*coeffs[22]*coeffs[19] + 2*coeffs[1]*coeffs[17]*coeffs[4]*coeffs[23] - 4*coeffs[1]*coeffs[21]*coeffs[2]*coeffs[23] + 2*coeffs[1]*coeffs[21]*coeffs[4]*coeffs[19] + 2*coeffs[3]*coeffs[17]*coeffs[2]*coeffs[23] - 4*coeffs[3]*coeffs[17]*coeffs[4]*coeffs[19] + 2*coeffs[3]*coeffs[21]*coeffs[2]*coeffs[19];
    coeffs[67] = 2*coeffs[17]*coeffs[20] + 2*coeffs[18]*coeffs[19];
    coeffs[68] = 2*coeffs[0]*coeffs[2]*coeffs[16]*coeffs[18] - 2*coeffs[17]*coeffs[4]*coeffs[4]*coeffs[19] - 2*coeffs[21]*coeffs[2]*coeffs[2]*coeffs[23] - 2*coeffs[0]*coeffs[0]*coeffs[18]*coeffs[20] - 2*coeffs[3]*coeffs[3]*coeffs[18]*coeffs[20] - 2*coeffs[1]*coeffs[1]*coeffs[22]*coeffs[24] - 2*coeffs[15]*coeffs[2]*coeffs[2]*coeffs[16] + 2*coeffs[0]*coeffs[15]*coeffs[2]*coeffs[20] + 2*coeffs[1]*coeffs[3]*coeffs[18]*coeffs[24] + 2*coeffs[1]*coeffs[3]*coeffs[22]*coeffs[20] + 2*coeffs[1]*coeffs[17]*coeffs[4]*coeffs[24] - 4*coeffs[1]*coeffs[21]*coeffs[2]*coeffs[24] + 2*coeffs[1]*coeffs[21]*coeffs[4]*coeffs[20] - 4*coeffs[1]*coeffs[2]*coeffs[22]*coeffs[23] + 2*coeffs[1]*coeffs[4]*coeffs[18]*coeffs[23] + 2*coeffs[1]*coeffs[4]*coeffs[22]*coeffs[19] + 2*coeffs[3]*coeffs[17]*coeffs[2]*coeffs[24] - 4*coeffs[3]*coeffs[17]*coeffs[4]*coeffs[20] + 2*coeffs[3]*coeffs[21]*coeffs[2]*coeffs[20] + 2*coeffs[3]*coeffs[2]*coeffs[18]*coeffs[23] + 2*coeffs[3]*coeffs[2]*coeffs[22]*coeffs[19] - 4*coeffs[3]*coeffs[4]*coeffs[18]*coeffs[19] + 2*coeffs[17]*coeffs[2]*coeffs[4]*coeffs[23] + 2*coeffs[21]*coeffs[2]*coeffs[4]*coeffs[19];
    coeffs[69] = 2*coeffs[18]*coeffs[20];
    coeffs[70] = - coeffs[0]*coeffs[0]*coeffs[19]*coeffs[19] + 2*coeffs[0]*coeffs[1]*coeffs[16]*coeffs[19] - coeffs[1]*coeffs[1]*coeffs[16]*coeffs[16] - coeffs[1]*coeffs[1]*coeffs[23]*coeffs[23] + 2*coeffs[1]*coeffs[3]*coeffs[19]*coeffs[23] - coeffs[3]*coeffs[3]*coeffs[19]*coeffs[19];
    coeffs[71] = coeffs[19]*coeffs[19];
    coeffs[72] = - 2*coeffs[20]*coeffs[0]*coeffs[0]*coeffs[19] + 2*coeffs[20]*coeffs[0]*coeffs[1]*coeffs[16] + 2*coeffs[2]*coeffs[0]*coeffs[16]*coeffs[19] - 2*coeffs[24]*coeffs[1]*coeffs[1]*coeffs[23] + 2*coeffs[24]*coeffs[1]*coeffs[3]*coeffs[19] + 2*coeffs[20]*coeffs[1]*coeffs[3]*coeffs[23] - 2*coeffs[2]*coeffs[1]*coeffs[16]*coeffs[16] + 2*coeffs[4]*coeffs[1]*coeffs[19]*coeffs[23] - 2*coeffs[2]*coeffs[1]*coeffs[23]*coeffs[23] - 2*coeffs[20]*coeffs[3]*coeffs[3]*coeffs[19] - 2*coeffs[4]*coeffs[3]*coeffs[19]*coeffs[19] + 2*coeffs[2]*coeffs[3]*coeffs[19]*coeffs[23];
    coeffs[73] = 2*coeffs[19]*coeffs[20];
    coeffs[74] = - coeffs[0]*coeffs[0]*coeffs[20]*coeffs[20] + 2*coeffs[0]*coeffs[2]*coeffs[16]*coeffs[20] - coeffs[1]*coeffs[1]*coeffs[24]*coeffs[24] + 2*coeffs[1]*coeffs[3]*coeffs[20]*coeffs[24] - 4*coeffs[1]*coeffs[2]*coeffs[23]*coeffs[24] + 2*coeffs[1]*coeffs[4]*coeffs[19]*coeffs[24] + 2*coeffs[1]*coeffs[4]*coeffs[23]*coeffs[20] - coeffs[3]*coeffs[3]*coeffs[20]*coeffs[20] + 2*coeffs[3]*coeffs[2]*coeffs[19]*coeffs[24] + 2*coeffs[3]*coeffs[2]*coeffs[23]*coeffs[20] - 4*coeffs[3]*coeffs[4]*coeffs[19]*coeffs[20] - coeffs[2]*coeffs[2]*coeffs[16]*coeffs[16] - coeffs[2]*coeffs[2]*coeffs[23]*coeffs[23] + 2*coeffs[2]*coeffs[4]*coeffs[19]*coeffs[23] - coeffs[4]*coeffs[4]*coeffs[19]*coeffs[19];
    coeffs[75] = coeffs[20]*coeffs[20];
    
    static const int coeffs0_ind[] = {25,26,27,28,  32,33,34,35,  40,41,42,43,44,45,  58,59,60,61,62,63,  25,26,27,28,  32,33,34,35,  25,26,27,28,  32,33,34,35,  25,26,27,28};
    static const int coeffs1_ind[] = {29,30,31,     36,37,38,39,  46,47,48,49,50,51,  64,65,66,67,68,69,  29,30,31,     36,37,38,39,  29,30,31,     36,37,38,39,  29,30,31};
    static const int coeffs2_ind[] = {52,53,54,55,56,57,  70,71,72,73,74,75};
    
    static const int C0_ind[] = {0,1,3,4,  9,10,12,13,  18,20,21,23,24,26,  27,29,30,32,33,35,  37,38,40,41, 46,47,49,50, 57,58,60,61, 66,67,69,70,  76,77,79,80};
    static const int C1_ind[] = {0,1,3,    9,10,12,13,  18,20,21,23,24,26,  27,29,30,32,33,35,  37,38,40,    46,47,49,50, 57,58,60,    66,67,69,70,  76,77,79};
    static const int C2_ind[] = {18,20,21,23,24,26,  27,29,30,32,33,35};
    
    Matrix<double, 9, 9, RowMajor> C0;
    C0.setZero();
    Matrix<double, 9, 9, RowMajor> C1;
    C1.setZero();
    Matrix<double, 9, 9, RowMajor> C2;
    C2.setZero();
    
    for (int i = 0; i < 40; i++)
    {
        C0(C0_ind[i]) = coeffs(coeffs0_ind[i]);
    }
    for (int i = 0; i < 36; i++)
    {
        C1(C1_ind[i]) = coeffs(coeffs1_ind[i]);
    }
    for (int i = 0; i < 12; i++)
    {
        C2(C2_ind[i]) = coeffs(coeffs2_ind[i]);
    }
    
      
    
    Matrix<double, 9, 9> A0 = C0.transpose();
    Matrix<double, 9, 9> A1 = C1.transpose();
    Matrix<double, 9, 2> A2 = C2.block<2, 9>(2, 0).transpose();
    
    
    
    MatrixXd M = MatrixXd::Zero(11, 11);
    MatrixXd Mm(9, 11);
    Mm << A2, A1;
    Mm = (-A0.partialPivLu().solve(Mm)).eval();
    
    M.block<9, 11>(2, 0) = Mm;
    
    M(0,4) = 1; M(1,5) = 1; 
    
    RealSchur<MatrixXd> schur(11);
    schur.compute(M, false);
    MatrixXd sols_o = MatrixXd::Zero(1, 11);
    MatrixXd S = schur.matrixT();
    int k = 0;
    for (int i = 0; i < 10; i++)
    {
        if ((S(i, i) > 0) && (S(i + 1, i) == 0))
        {
            sols_o(0, k) = S(i, i);
            k++;
        }
    }
    if ((S(10, 10) > 0) && (S(9, 10) == 0))
    {
        sols_o(0, k) = S(10, 10);
        k++;
    }
    
    
    MatrixXd U = MatrixXd::Zero(9, 9);
    MatrixXd V = MatrixXd::Zero(8, 1);
    
    MatrixXd sols = MatrixXd::Zero(3, k);
    int m =0;
    double f;
    
    for (int i = 0; i < k; i++)
    {
        f = 1/sols_o(0,i);
        U = f*f*C2 + f*C1 + C0;
        V = -U.block<8, 8>(0, 1).partialPivLu().solve(U.block<8, 1>(0, 0));
        if (  (V(2,0)>0) &&  (fabs(V(5,0)-V(2,0)*V(2,0)) < 0.01*V(5,0)))
        {
            sols(0,m) = f; // 1st row - f2, 2nd row - 1/f1, 3rd row - k.
            sols(1,m) = V(2,0);
            sols(2,m) = V(0,0);
            m++;
        }
        
    }
    
    sols.conservativeResize(3,m);
    return sols;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const VectorXd data = Map<const VectorXd>(mxGetPr(prhs[0]), 23);
    MatrixXd sols = solver_var(data);
    plhs[0] = mxCreateDoubleMatrix(sols.rows(), sols.cols(), mxREAL);
    double *zr = mxGetPr(plhs[0]);
    for (Index i = 0; i < sols.size(); i++)
    {
        zr[i] = sols(i);
    }
}
