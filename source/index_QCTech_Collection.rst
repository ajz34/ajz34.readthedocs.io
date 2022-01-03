合集 分子的电子结构方法与算法
=============================

是否有可能在一个软件中，高效且漂亮地包含所有电子结构的瑰宝？我想很难。但是否能用强大的前人所提供的有力工具，在较为统一的程序、文档框架下，帮助自己理解这复杂体系的林林总总？或许可以试试。

我会尝试以 PySCF 为主力工具，对现有的一些电子结构方法或算法作补充说明。这些内容未必是基础或常用的；我们不会涉及基础教科书中出现的内容，譬如简单基组、Hartree-Fock 方法等。

这里所指的方法可能是一些 post-HF 方法 (譬如不常见的 CEPA, OO-MP2 等)，也可能是矫正方法 (譬如 DFT-D3, 溶剂化等)，或是密度泛函的一些贡献分量 (譬如 LT-SOS, VV10 等)。这里所指的算法可能是收敛方法 (譬如 DIIS)，也可能是加速算法 (RI, THC, COSX 等)。总之取材应会很宽泛。这本身也是我自己的随笔，如果哪天想起来，也会往里随便翻翻。

.. toctree::
   :maxdepth: 1
   :numbered:
   
   ../QC_Notes/PUHF_and_PMP2
   ../QC_Notes/Post_Series/mp3_mp4_energy
   ../QC_Notes/Post_Series/scsRPA_Comprehense
   ../QC_Notes/Post_Series/oomp2
   ../QC_Notes/Post_Series/cepa_learn
   ../QC_Notes/DF_Series/DF_SCF
   ../QC_Notes/DF_Series/DF_MP2
   ../QC_Notes/DF_Series/LT_MP2
   ../QC_Notes/SCF_Series/diis_comprehen
   ../QC_Notes/DF_Series/LS_THC_MP2
