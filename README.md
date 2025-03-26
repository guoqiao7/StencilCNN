由于服务器重装，cudnn丢失，新配置的cudnn位于 ~/cuda/{include,lib64}
nvcc conv.cu -o conv_without_cpu -arch=sm_80 -L ~/cuda/lib64 -lcudnn
