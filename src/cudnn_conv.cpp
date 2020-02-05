#include <iostream>
#include <memory>
#include "basic.h"
#include "cudnn_conv.h"

using namespace std;
using perf_t = cudnnConvolutionFwdAlgoPerf_t;

/**************************************************
CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3,
CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4,
CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5,
CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6,
CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED     = 7,
CUDNN_CONVOLUTION_FWD_ALGO_COUNT                 = 8
***************************************************/

bool int8_ext = true;
cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
//cudnnMathType_t math_type = CUDNN_DEFAULT_MATH;

const size_t kMaxCudnnWorkspace = 1024 * 1024 * 1024;
int kMaxDataSize = 512 * 512 * 128 * 4 * 2;
int kMaxWeightSize = 7 * 7 * 256 * 256 * 3 * 30;
cudnnHandle_t cudnn_handle_g;

CudnnConv::CudnnConv(int in_n, int in_c, int in_h, int in_w,
              int k_n, int k_c, int k_h, int k_w,
              int p_h, int p_w,
              int s_h, int s_w,
              int d_h, int d_w,
              int group,
              cudnnDataType_t in_type,
              cudnnDataType_t weight_type,
              cudnnDataType_t out_type,
              cudnnTensorFormat_t in_format,
              cudnnTensorFormat_t weight_format,
              cudnnTensorFormat_t output_format)
        : input_n_(in_n),input_c_(in_c), input_h_(in_h), input_w_(in_w),
          kernel_n_(k_n), kernel_c_(k_c), kernel_h_(k_h), kernel_w_(k_w),
          pad_h_(p_h), pad_w_(p_w),
          stride_h_(s_h), stride_w_(s_w),
          dilation_h_(d_h), dilation_w_(d_w),
          group_(group),
          input_type_(in_type), weight_type_(weight_type), output_type_(out_type),
          input_format_(in_format), weight_format_(weight_format), output_format_(output_format),
          input_data_(nullptr),
          weight_data_(nullptr),
          output_data_(nullptr),
          cudnn_workspace_(nullptr){
        CHECK_EXIT(group_ != 1, "only support group == 1 now");
        //CHECK_EXIT(in_c != kernel_c_, "in_c != kernel_c_");
        cudnnStatus_t sts;
        sts = cudnnCreateTensorDescriptor(&input_desc_);
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnCreateTensorDescriptor");
        sts = cudnnCreateTensorDescriptor(&output_desc_);
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnCreateTensorDescriptor");
        sts = cudnnCreateFilterDescriptor(&weight_desc_);
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnCreateTensorDescriptor");
        sts =cudnnCreateConvolutionDescriptor(&conv_desc_);
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnCreateConvolutionDescriptor");

#if 0
        sts = cudnnSetTensor4dDescriptorEx(input_desc_,
                                           input_type_,
                                           input_n_,
                                           input_c_,
                                           input_h_,
                                           input_w_,
                                           input_c_ * input_h_ * input_w_,
                                           input_h_ * input_w_,
                                           input_w_,
                                           1);
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnSetTensor4dDescriptorEx");
        sts = cudnnSetTensor4dDescriptorEx(output_desc_,
                                           output_type_,
                                           output_n(),
                                           output_c(),
                                           output_h(),
                                           output_w(),
                                           output_c() * output_h() * output_w(),
                                           output_h() * output_w(),
                                           output_w(),
                                           1);
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnSetTensor4dDescriptorEx");
#endif
#if 1
        sts = cudnnSetTensor4dDescriptor(input_desc_,
                                         input_format_,
                                         input_type_,
                                         input_n_,
                                         input_c_,
                                         input_h_,
                                         input_w_);

        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnSetTensor4dDescriptorEx");
        
        sts = cudnnSetTensor4dDescriptor(output_desc_,
                                         output_format_,
                                         output_type_,
                                         output_n(),
                                         output_c(),
                                         output_h(),
                                         output_w());
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnSetTensor4dDescriptorEx");
#endif
        cudnnDataType_t conv_t = conv_type();
        sts = cudnnSetConvolution2dDescriptor(conv_desc_, 
                                              pad_h_,
                                              pad_w_,
                                              stride_h_,
                                              stride_w_,
                                              dilation_h_,
                                              dilation_w_,
                                              CUDNN_CROSS_CORRELATION,
                                              conv_t);
        
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnSetConvolution2dDescriptor");
        sts = cudnnSetFilter4dDescriptor(weight_desc_, 
                                         weight_type_,
                                         weight_format_,
                                         kernel_n_,
                                         kernel_c_,
                                         kernel_h_,
                                         kernel_w_);
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnSetFilter4dDescriptor");
}

cudnnDataType_t CudnnConv::conv_type() {
    if ((input_type_ == CUDNN_DATA_FLOAT) &&
        (output_type_ == CUDNN_DATA_FLOAT) &&
        (weight_type_ == CUDNN_DATA_FLOAT)) {
        return CUDNN_DATA_FLOAT;
    } else if ((input_type_ == CUDNN_DATA_HALF) &&
            (output_type_ == CUDNN_DATA_HALF) && 
             (weight_type_ == CUDNN_DATA_HALF)) {
        //return CUDNN_DATA_FLOAT;
        return CUDNN_DATA_HALF;
    } else if ((input_type_ == CUDNN_DATA_INT8) &&
            (output_type_ == CUDNN_DATA_INT8) && 
             (weight_type_ == CUDNN_DATA_INT8)) {
        return CUDNN_DATA_INT32;
        //return CUDNN_DATA_INT8x32;
        //return CUDNN_DATA_INT8;
    } else if ((input_type_ == CUDNN_DATA_INT8x4) &&
            (output_type_ == CUDNN_DATA_INT8x4) && 
             (weight_type_ == CUDNN_DATA_INT8x4)) {
        //return CUDNN_DATA_FLOAT;
        return CUDNN_DATA_INT32;
    } else if ((input_type_ == CUDNN_DATA_INT8x32) &&
            (output_type_ == CUDNN_DATA_INT8x32) && 
             (weight_type_ == CUDNN_DATA_INT8x32)) {
        return CUDNN_DATA_INT32;
    } else {
        CHECK_EXIT(true, "conv_type not support");
    }
}

const char* conv_type_str[] = {
 "CUDNN_DATA_FLOAT   ",
 "CUDNN_DATA_DOUBLE  ",
 "CUDNN_DATA_HALF    ",
 "CUDNN_DATA_INT8    ",
 "CUDNN_DATA_INT32   ",
 "CUDNN_DATA_INT8x4  ",
 "CUDNN_DATA_UINT8   ",
 "CUDNN_DATA_UINT8x4 ",
 "CUDNN_DATA_INT8x32 ",
};

void CudnnConv::InitAlgo(cudnnHandle_t handle) {
    cudnnStatus_t sts;
    sts = cudnnSetConvolutionMathType(conv_desc_, math_type);
    CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnSetConvolutionMathType");
    //cout << "conv_type: " << conv_type_str[conv_type()] << endl;

#if 0
	int perf_count;
	static constexpr int num_algos = 8;
 	std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
	size_t work_size = 10 * 1024 * 1024;
	void* work_data;
	cudaMalloc(&work_data, work_size);
	
	cudnnFindConvolutionForwardAlgorithmEx(handle,
			 input_desc_,
		     input_data_g, 
	         weight_desc_,
			 weight_data_g,
			 conv_desc_,
		     output_desc_,
			 output_data_g,
	         num_algos,
	         &perf_count,
	         perf_results.get(),
	         work_data,
	         work_size);
    cudaFree(work_data);
  	algo_ = perf_results[0].algo;

#else
    if (conv_type() == CUDNN_DATA_INT32) {
    //if (false) {
        // algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        //algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

        sts = cudnnGetConvolutionForwardAlgorithm(handle, 
                                                  input_desc_,
                                                  weight_desc_,
                                                  conv_desc_,
                                                  output_desc_,
                                                  CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                  0,
                                                  &algo_);
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnGetConvolutionForwardAlgorithm");
        //CHECK_EXIT(algo_ != CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, "algo error");
    //} else if (conv_type() == CUDNN_DATA_HALF) {
    //    algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    } else {
        sts = cudnnGetConvolutionForwardAlgorithm(handle, 
                                                  input_desc_,
                                                  weight_desc_,
                                                  conv_desc_,
                                                  output_desc_,
                                                  CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                  0,
                                                  &algo_);
        CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnGetConvolutionForwardAlgorithm");
        //algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    }
#endif

    //algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    //cout << "algo_: " << algo_ << endl;
    sts = cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                  input_desc_,
                                                  weight_desc_,
                                                  conv_desc_,
                                                  output_desc_,
                                                  algo_,
                                                  &cudnn_workspace_size_);
    //printf("cudnn workspace size: %d, max cudnn waorkspace: %d\n", cudnn_workspace_size_, kMaxCudnnWorkspace);
    CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnGetConvolutionForwardWorkspaceSize");
    CHECK_EXIT(cudnn_workspace_size_ > kMaxCudnnWorkspace, "cudnn_workspace_size_ > kMaxCudnnWorkspace");
}

void CudnnConv::Run(void* input,
                    void* weight,
                    void* output,
                    void* cudnn_workspace,
                    cudnnHandle_t handle) {
    float alpha = 1.0f;
    float beta = 1.0f;
    cudnnStatus_t sts;
    
    
    sts = cudnnSetConvolutionMathType(conv_desc_, math_type);
    CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnSetConvolutionMathType");
    
    sts = cudnnConvolutionForward(handle,
                                  &alpha,
                                  input_desc_,
                                  input,
                                  weight_desc_,
                                  weight,
                                  conv_desc_,
                                  algo_,
                                  cudnn_workspace,
                                  cudnn_workspace_size_,
                                  &beta,
                                  output_desc_,
                                  output);
    CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnConvolutionForward");
}
