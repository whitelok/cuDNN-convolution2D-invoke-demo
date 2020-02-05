#ifndef _PROFILE_H_
#define _PROFILE_H_

#define LINUX
#define OPT_CUDA_PROFILE
#define OPT_TIMER

#ifdef OPT_CUDA_PROFILE
#include <cuda_runtime.h>
#endif

#ifdef LINUX
#include <sys/time.h>
#else
#include <windows.h>
#endif

#define     MAX_OPT_PROFILE_INFO_NUM        1024            //���profile����

#ifdef  OPT_TIMER           //��profile����
    
//profile��Ϣ�ṹ��
struct OPT_PROFILE_INFO
{
#ifdef LINUX
    struct timeval start, end;
#else
    LARGE_INTEGER start, end, freq;
#endif

#ifdef OPT_CUDA_PROFILE
    cudaEvent_t     e_start;
    cudaEvent_t     e_stop;
#endif

    double          time_used;                              //��ʱ
    int             profile_count;                          //��������
    double          time_used_total;                        //�ܺ�ʱ
    double          calculation;                            //������
    double          gflops;                                 //������ / ��ʱ

    double          time_used_max;
    double          time_used_min;

};

//����ȫ�ֱ���
extern      OPT_PROFILE_INFO                opt_profile_info[];



#define         OPT_PROFILE_TIME_START(num)                         opt_profile_time_start(num)                        
#define         OPT_PROFILE_TIME_STOP(num, str, iter, print_flag)           opt_profile_time_stop(num, str, iter, print_flag)
#define         OPT_PROFILE_TIME_RESET(num)                         opt_profile_time_reset(num)
#define         OPT_PRINT_INFO(str, num)                            opt_print_info(str, num)  
    
/***************************************************************************************************
* ��  ��: profile��ʱ��Ϣ
* ��  ��:
*               num         ָ��ʹ�õļ�ʱ��
* ����ֵ: ��
***************************************************************************************************/
void opt_profile_time_start(int num);


/***************************************************************************************************
* ��  ��: profile��ʱ��Ϣ
* ��  ��:
*               num                 ָ��ʹ�õļ�ʱ��
*               str                 �ַ�����Ϣ
*               iter                �ܹ���������
*               print_flag          ��ӡ��Ϣ��־
* ����ֵ: ��
***************************************************************************************************/
void opt_profile_time_stop(int num, const char *str, int iter, int print_flag);

void opt_print_info(const char *str, int num);

void opt_profile_time_reset(int num);

#else           //�ر�profile����


#define         OPT_PROFILE_TIME_START(num)                                              //profile��ʱ��Ϣ��
#define         OPT_PROFILE_TIME_STOP(num, str, iter, print_flag)                        //profile��ʱ��Ϣ��
#define         OPT_PROFILE_TIME_RESET(num)
#define         OPT_PRINT_INFO(num)

#endif



#endif
