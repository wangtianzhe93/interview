# torch

## torch memory

- **stride**
    - Reference: [https://martinlwx.github.io/zh-cn/how-to-reprensent-a-tensor-or-ndarray/](https://martinlwx.github.io/zh-cn/how-to-reprensent-a-tensor-or-ndarray/)
    
    - Torch tensor storage
        - 在torch里, 一个二维数组中一般在内存里占据连续的位置, 但是按行和按列存储是不同的. 比如现在有一个`shape=[2, 3]`的二维数组:
        ```python
            [[0.2949, 0.9608, 0.0965],
            [0.5463, 0.4176, 0.8146]]
        ```
        如果是按行存储, 那么内存中的排列(`A_in_row`)是:
        ```python
            [0.2949, 0.9608, 0.0965, 0.5463, 0.4176, 0.8146]
        ```
        按行存储时, 要访问`(i, j)`位置的值可以通过:
        ```python
            A[i][j] = A_in_row[i * A.shape[1] + j]
        ```
        如果是按列存储, 那么内存中的排列(这里记为`A_in_col`)是:
        ```
            [0.2949, 0.5463, 0.9608, 0.4176, 0.0965, 0.8146]
        ```
        按列存储时, 要访问`(i, j)`位置的值可以通过:
        ```python
            A[i][j] = A_in_col[j * A.shape[0] + i]
        ```

    - stride
        - tensor在内存中可以按照行或者列进行存储, numpy和pytorch默认都是按照行进行存储的, 其实在内存中, 任何维度的tensor的底层存储都会在内存中占据连续的空间, 所以访问内存的方式就变得很重要, 在pytorch里是通过stride访问的.
        - stride可以看作是前面按行和按列两种格式索引的泛化版本. 假设现在有一个N维的tensor A, 在内存中的存储为`A_internal`, 想要访问`A[i0][i1][i2] ...`, 需要索引方式如下:
        ```python
        A[i0][i1][i2] ... = A_internal[
            stride_offset
            + i0 * A.strides[0]
            + i1 * A.strides[1]
            + i2 * A.strides[2]
            + ...
            + in-1 * A.strides[n-1]
        ]
        ```
        - strides格式有两个组成部分
            - offset: 表示张量相对于底层存储`A_internal`的偏移量.
            - strides: 是一个数组, 长度和张量的维度一样, `strides[i]`表示张量在第`i`个维度上移动"一个单位"需要在内存上跳过多少个元素.
        - 例如, 前面提到的二维数组的例子, 如果用strides的格式来理解的话，应该是下面这样:
        ```python
            A[i][j] = A_in_row[
                0
                + i * A.shape[1]
                + j * 1
            ]
        ```
        - 对于一个大小为`(A.shape[0], A.shape[1])`的二维数组, 它的offset是0, `strides = [A.shape[1], 1]`(row-major). 也就是说, 每次在第一个维度上要跳跃"一个单位", 需要跳过底层的`A.shape[1]`个元素, `A.shape[1]`即为行的长度. 即下图所示.

        ![Untitled](torch/torch1.png)

        - 那么如何得到`N`维张量的`strides`数组? 假设要求解的是`strides[k]`即第`k`个维度的`stride`, 我们知道它的语义是"在第`k`维上移动一个单位需要在内存上跳过多少个元素", 如果这个张量的底层存储在内存上是连续存储(紧凑格式), 那就是`k+1, k+2, ... , N-1`维度的大小的乘积, 如果 `k = N - 1`, 那么`strides[N - 1] = 1`.
        - 数学公式如下:
            $strides[k]=\prod_{i=k+1}^{N-1}shape[i]$
        - 上面的公式只有在张量的底层存储在内存上是连续存储(紧凑格式)的时候成立.

    - Strides常见操作
        - 使用stride之后, 很多关于tensor的操作都可以是零拷贝(Zero-copy)的. 通过strides格式, "底层存储"和"视图"之间是分离开的.
        - print_internal
            - 首先, 写一个函数获取Pytorch的tensor底层存储. 使用Pytorch提供的`data_ptr()`, 会返回tensor底层存储表示的第一个元素的内存地址.
            - 然后通过Pytorch提供的`storage().nbytes()`就可以知道当前tensor的底层存储在内存上占据了多大的空间, 而tensor的`dtype`属性则告诉了我们每个元素占据了多大的空间, 比如`torch.float32`就是4个字节.
            - 最后通过`ctypes.string_at(address, size=-1)`函数就可以读取这个张量为`C`的字符串(buffer), 而`torch.frombuffer`可以从一个 buffer创建出tensor.
            - 通过上面几个步骤，我们就可以还原出Pytorch底层的数组表示，下面命名为`print_internal`函数:
            ```python
            def print_internal(t: torch.Tensor):
                print(
                    torch.frombuffer(
                        ctypes.string_at(t.data_ptr(), t.storage().nbytes()), dtype=t.dtype
                    )
                )
            ```
            - 然后我们创建一个维度为(1, 2, 3, 4)的tensor t并观察它的底层表示, 后面的操作讲解会基于这个tensor:
            ```python
                t = torch.arange(0, 24).reshape(1, 2, 3, 4)
                print(t)
                # tensor([[[[ 0,  1,  2,  3],
                #           [ 4,  5,  6,  7],
                #           [ 8,  9, 10, 11]],

                #          [[12, 13, 14, 15],
                #           [16, 17, 18, 19],
                #           [20, 21, 22, 23]]]])

                print(t.stride())
                # (24, 12, 4, 1)

                print_internal(t)
                # tensor([0,  1,  2,  3,
                #         4,  5,  6,  7,
                #         8,  9, 10, 11,
                #         12, 13, 14, 15,
                #         16, 17, 18, 19,
                #         20, 21, 22, 23])
            ```
            - 按照我们前面说的从张量的维度推导stride的方法, 我们不难知道这个tensor的stride应该是`(2 * 3 * 4, 3 * 4, 4, 1)`也就是`(24, 12, 4, 1)`. 在 Pytorch 里面，我们可以通过调用 tensor 的 stride() 方法获得 stride，可以看到，确实跟我们手动计算出来的一样.

