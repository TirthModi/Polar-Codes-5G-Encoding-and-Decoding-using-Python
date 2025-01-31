"""
    Lab Group: 6
    Project Group: 3
    Topic: Polar Codes
    Sub-topic: Polar Code SC vs SCL Decoding
"""
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import random as rd
import scipy.stats as stats
from scipy.stats import norm
import galois as gf
from scipy.special import erfc

print()
Reliability_Sequence = [  0,   1,   2,   4,   8,  16,  32,   3,   5,  64,   9,   6,  17,  10,  18, 128,  12,  33,  65,  20,
                        256,  34,  24,  36,   7, 129,  66, 512,  11,  40,  68, 130,  19,  13,  48,  14,  72, 257,  21, 132,
                         35, 258,  26, 513,  80,  37,  25,  22, 136, 260, 264,  38, 514,  96,  67,  41, 144,  28,  69,  42,
                        516,  49,  74, 272, 160, 520, 288, 528, 192, 544,  70,  44, 131,  81,  50,  73,  15, 320, 133,  52, 
                         23, 134, 384,  76, 137,  82,  56,  27,  97,  39, 259,  84, 138, 145, 261,  29,  43,  98, 515,  88, 
                        140,  30, 146,  71, 262, 265, 161, 576,  45, 100, 640,  51, 148,  46,  75, 266, 273, 517, 104, 162,  
                         53, 193, 152,  77, 164, 768, 268, 274, 518,  54,  83,  57, 521, 112, 135,  78, 289, 194,  85, 276, 
                        522,  58, 168, 139,  99,  86,  60, 280,  89, 290, 529, 524, 196, 141, 101, 147, 176, 142, 530, 321,  
                         31, 200,  90, 545, 292, 322, 532, 263, 149, 102, 105, 304, 296, 163,  92,  47, 267, 385, 546, 324, 
                        208, 386, 150, 153, 165, 106,  55, 328, 536, 577, 548, 113, 154,  79, 269, 108, 578, 224, 166, 519, 
                        552, 195, 270, 641, 523, 275, 580, 291,  59, 169, 560, 114, 277, 156,  87, 197, 116, 170,  61, 531, 
                        525, 642, 281, 278, 526, 177, 293, 388,  91, 584, 769, 198, 172, 120, 201, 336,  62, 282, 143, 103, 
                        178, 294,  93, 644, 202, 592, 323, 392, 297, 770, 107, 180, 151, 209, 284, 648,  94, 204, 298, 400, 
                        608, 352, 325, 533, 155, 210, 305, 547, 300, 109, 184, 534, 537, 115, 167, 225, 326, 306, 772, 157,
                        656, 329, 110, 117, 212, 171, 776, 330, 226, 549, 538, 387, 308, 216, 416, 271, 279, 158, 337, 550, 
                        672, 118, 332, 579, 540, 389, 173, 121, 553, 199, 784, 179, 228, 338, 312, 704, 390, 174, 554, 581, 
                        393, 283, 122, 448, 353, 561, 203,  63, 340, 394, 527, 582, 556, 181, 295, 285, 232, 124, 205, 182, 
                        643, 562, 286, 585, 299, 354, 211, 401, 185, 396, 344, 586, 645, 593, 535, 240, 206,  95, 327, 564, 
                        800, 402, 356, 307, 301, 417, 213, 568, 832, 588, 186, 646, 404, 227, 896, 594, 418, 302, 649, 771, 
                        360, 539, 111, 331, 214, 309, 188, 449, 217, 408, 609, 596, 551, 650, 229, 159, 420, 310, 541, 773, 
                        610, 657, 333, 119, 600, 339, 218, 368, 652, 230, 391, 313, 450, 542, 334, 233, 555, 774, 175, 123, 
                        658, 612, 341, 777, 220, 314, 424, 395, 673, 583, 355, 287, 183, 234, 125, 557, 660, 616, 342, 316, 
                        241, 778, 563, 345, 452, 397, 403, 207, 674, 558, 785, 432, 357, 187, 236, 664, 624, 587, 780, 705, 
                        126, 242, 565, 398, 346, 456, 358, 405, 303, 569, 244, 595, 189, 566, 676, 361, 706, 589, 215, 786, 
                        647, 348, 419, 406, 464, 680, 801, 362, 590, 409, 570, 788, 597, 572, 219, 311, 708, 598, 601, 651, 
                        421, 792, 802, 611, 602, 410, 231, 688, 653, 248, 369, 190, 364, 654, 659, 335, 480, 315, 221, 370, 
                        613, 422, 425, 451, 614, 543, 235, 412, 343, 372, 775, 317, 222, 426, 453, 237, 559, 833, 804, 712, 
                        834, 661, 808, 779, 617, 604, 433, 720, 816, 836, 347, 897, 243, 662, 454, 318, 675, 618, 898, 781, 
                        376, 428, 665, 736, 567, 840, 625, 238, 359, 457, 399, 787, 591, 678, 434, 677, 349, 245, 458, 666, 
                        620, 363, 127, 191, 782, 407, 436, 626, 571, 465, 681, 246, 707, 350, 599, 668, 790, 460, 249, 682, 
                        573, 411, 803, 789, 709, 365, 440, 628, 689, 374, 423, 466, 793, 250, 371, 481, 574, 413, 603, 366, 
                        468, 655, 900, 805, 615, 684, 710, 429, 794, 252, 373, 605, 848, 690, 713, 632, 482, 806, 427, 904, 
                        414, 223, 663, 692, 835, 619, 472, 455, 796, 809, 714, 721, 837, 716, 864, 810, 606, 912, 722, 696, 
                        377, 435, 817, 319, 621, 812, 484, 430, 838, 667, 488, 239, 378, 459, 622, 627, 437, 380, 818, 461, 
                        496, 669, 679, 724, 841, 629, 351, 467, 438, 737, 251, 462, 442, 441, 469, 247, 683, 842, 738, 899, 
                        670, 783, 849, 820, 728, 928, 791, 367, 901, 630, 685, 844, 633, 711, 253, 691, 824, 902, 686, 740, 
                        850, 375, 444, 470, 483, 415, 485, 905, 795, 473, 634, 744, 852, 960, 865, 693, 797, 906, 715, 807, 
                        474, 636, 694, 254, 717, 575, 913, 798, 811, 379, 697, 431, 607, 489, 866, 723, 486, 908, 718, 813, 
                        476, 856, 839, 725, 698, 914, 752, 868, 819, 814, 439, 929, 490, 623, 671, 739, 916, 463, 843, 381, 
                        497, 930, 821, 726, 961, 872, 492, 631, 729, 700, 443, 741, 845, 920, 382, 822, 851, 730, 498, 880, 
                        742, 445, 471, 635, 932, 687, 903, 825, 500, 846, 745, 826, 732, 446, 962, 936, 475, 853, 867, 637, 
                        907, 487, 695, 746, 828, 753, 854, 857, 504, 799, 255, 964, 909, 719, 477, 915, 638, 748, 944, 869, 
                        491, 699, 754, 858, 478, 968, 383, 910, 815, 976, 870, 917, 727, 493, 873, 701, 931, 756, 860, 499, 
                        731, 823, 922, 874, 918, 502, 933, 743, 760, 881, 494, 702, 921, 501, 876, 847, 992, 447, 733, 827, 
                        934, 882, 937, 963, 747, 505, 855, 924, 734, 829, 965, 938, 884, 506, 749, 945, 966, 755, 859, 940, 
                        830, 911, 871, 639, 888, 479, 946, 750, 969, 508, 861, 757, 970, 919, 875, 862, 758, 948, 977, 923, 
                        972, 761, 877, 952, 495, 703, 935, 978, 883, 762, 503, 925, 878, 735, 993, 885, 939, 994, 980, 926, 
                        764, 941, 967, 886, 831, 947, 507, 889, 984, 751, 942, 996, 971, 890, 509, 949, 973, 1000, 892, 950, 
                        863, 759, 1008, 510, 979, 953, 763, 974, 954, 879, 981, 982, 927, 995, 765, 956, 887, 985, 997, 986, 
                        943, 891, 998, 766, 511, 988, 1001, 951, 1002, 893, 975, 894, 1009, 955, 1004, 1010, 957, 983, 958, 
                        987, 1012, 999, 1016, 767, 989, 1003, 990, 1005, 959, 1011, 1013, 895, 1006, 1014, 1017, 1018, 991, 
                        1020, 1007, 1015, 1019, 1021, 1022, 1023]
 
# Function to set N-K frozen positions to zero and remaining K positions to message bits as per Reliability Sequence
def getUSequence(N, K, message, rs):
    
    u_sequence = np.zeros(N, dtype=int)
    i, j = N-K, 0
    while i < N:
        u_sequence[rs[i]] = message[j]
        j += 1
        i += 1
    # end
    return u_sequence

# An array representing state bit position like frozen bit or not (if not frozen then message bit).
def Frozen(rs, N, K):
    isFrozen = [False] * N
    for i in range(N-K):
        isFrozen[rs[i]] = True
    return isFrozen

# Polar Transform G2 or Knonecker Product for N=2
def PolarTransfromG2(u0, u1):

    n = len(u0)
    x = [0] * (2*n)
    
    for i in range(n):
        x[i] = (u0[i] + u1[i]) % 2
    
    for i in range(n, 2*n):
        x[i] = u1[i-n]

    return x

# Min-Sum Approximation : f(L1, L2) ~= sign(L1) * sign(L2) * min(|L1|, |L2|)
def minsum(L1, L2):
    
    n = len(L1)
    min_sum = [0] * n
    for i in range(n):
        min_sum[i] = np.sign(L1[i]) * np.sign(L2[i]) * min(abs(L1[i]), abs(L2[i]))
    return min_sum

# G Function for simple : g(L1, L2, b) = L2 + (1 - 2b) * L1 
def g(L1, L2, b):

    n = len(L1)
    rep_code = [0] * n
    for i in range(n):
        rep_code[i] = L2[i] + (1 - 2*b[i]) * L1[i]
    return rep_code

# Function two check wether the transmitted message is same as received message or not
def isEqual(tMsg, rMsg):

    n = len(tMsg)
    not_equal = False
    bit_err = 0
    for i in range(n):
        if tMsg[i] != rMsg[i]:
            not_equal = True
            bit_err += 1
    return (not not_equal, bit_err/n)

def encode(u_sequence, n, N):

    N = len(u_sequence)
    encoded_message = np.copy(u_sequence)
    m = 1
    for depth in range(n-1, -1, -1):
        for idx in range(0, N, 2*m):

            u0 = encoded_message[idx: idx+m]
            u1 = encoded_message[idx+m: idx+2*m]

            # u0 = [u0 + u1, u1]
            encoded_message[idx: idx+m] = (u0 + u1) % 2
            encoded_message[idx+m: idx+2*m] = u1
        # End
        m = m * 2
    # End
    return encoded_message

# Binary Phase Shift Keying (BPSK)
def BPSK(message):
    return 1 - 2 * message

# Additive White Gaussian Noise (AWGN)
def AWGN(message, Sigma):
    # return message + Sigma * np.random.randn(N)
    noise = []
    n = len(message)
    for i in range(n):
        noise.append(Sigma * rd.normalvariate(0, 1))
    return message + noise

# Time Complexity: NLogN
def decode_sc(L, node, depth, isFrozen):

    # We are at leaf node
    if depth == 0:

        # If given bit is frozen then simply return 0
        if isFrozen[node]:
            return [[0], [0]]
        else:
            # If L >= 0 then make hard decision that bit is 0 else it is 1
            if L[0] >= 0:
                return [[0], [0]]
            else:
                return [[1], [1]]
            # End hard decision
    # End leaf

    # L = [r1 r2]
    n = len(L)
    mid = n // 2
    r1 = L[0 : mid]
    r2 = L[mid : n]

    # Step L: Decode for the left child
    left_child_beliefs = minsum(r1, r2)
    u_cap_combined_left, u_cap_left = decode_sc(left_child_beliefs, 2*node, depth-1, isFrozen)

    # Step R: Decode for the right child
    right_child_beliefs = g(r1, r2, u_cap_combined_left)
    u_cap_combined_right, u_cap_right = decode_sc(right_child_beliefs, 2*node + 1, depth-1, isFrozen)

    # Step U: Combine left and right child decoded bits
    enc_msg = PolarTransfromG2(u_cap_combined_left, u_cap_combined_right)
    
    # Step End: Just combine decoded bits into a single list
    u_cap = []
    for u in u_cap_left:
        u_cap.append(u)

    for u in u_cap_right:
        u_cap.append(u)

    return [enc_msg, u_cap]
# End decoding

def decode_scl(L, node, depth, SCL_N, isFrozen):

    # We are at leaf node
    if depth == 0:

        decisions = []
        # If given bit is frozen then simply return 0
        if isFrozen[node]:
            if L[0] < 0:
                decisions.append([[0], [0], abs(L[0])])
            else:
                decisions.append([[0], [0], 0])
        else:
            # If L >= 0 then make hard decision that bit is 0 else it is 1
            if L[0] >= 0:
                decisions.append([[0], [0], 0])
                decisions.append([[1], [1], abs(L[0])])
            else:
                decisions.append([[1], [1], 0])
                decisions.append([[0], [0], abs(L[0])])
            # End hard decision
        return decisions
    # End leaf

    # L = [r1 r2]
    n = len(L)
    mid = n // 2
    r1 = L[0 : mid]
    r2 = L[mid : n]

    # Step L: Decode for the left child
    left_child_beliefs = minsum(r1, r2)
    left_child_decisions = decode_scl(left_child_beliefs, 2*node, depth-1, SCL_N, isFrozen)

    final_decisions = []
    # Step R: Decode for the right child
    for lch_decision in left_child_decisions:

        right_child_beliefs = g(r1, r2, lch_decision[0])
        right_child_decisions = decode_scl(right_child_beliefs, 2*node + 1, depth-1, SCL_N, isFrozen)

        # For every left combination perform Step R
        for rch_decision in right_child_decisions:

            # Step U: Combine left and right child decoded bits
            enc_msg = PolarTransfromG2(lch_decision[0], rch_decision[0])

            # Step End: Just combine decoded bits into a single list
            u_cap = []
            for u in lch_decision[1]:
                u_cap.append(u)

            for u in rch_decision[1]:
                u_cap.append(u)

            path_metric = lch_decision[2] + rch_decision[2]
            final_decisions.append([enc_msg, u_cap, path_metric])

    # Prunning the unnecessary branches
    final_decisions.sort(key=lambda x: x[2])
    final_decisions = sorted(final_decisions, key=lambda x: x[2])
    if len(final_decisions) > SCL_N:
        for i in range(SCL_N, len(final_decisions)):
            final_decisions.pop()
    
    return final_decisions
# End decoding


def sc_vs_scl(N, K, M):

    n = int(mt.log2(N))

    # Extracting Reliability Sequence for N bits
    rs = np.array([rs_pos for rs_pos in Reliability_Sequence if rs_pos < N])
    isFrozen = Frozen(rs, N, K)

    EbNodB_values = np.arange(0, 10, 0.5)
    P_success_sc = []
    P_success_scl = []
    BER_simulated_sc = []
    BER_simulated_scl = []
    BER_Theoretical = []

    # BER_Theoretical =  qfunc(np.sqrt(2 * Rate * EbNodB_values))
        
    for EbNodB in EbNodB_values:
        
        Success_sc = 0
        Success_scl = 0
        N_sim = 5000
        N_bit_errors_sc = 0
        N_bit_errors_scl = 0

        for sim in range(N_sim):

            Rate = K / N
            EbNo = 10 ** (EbNodB / 10)
            Sigma = mt.sqrt(1 / (Rate * EbNo))

            # Generating random K bits message
            message = np.random.randint(low = 0, high = 2, size = K, dtype = int)

            # Setting N-K frozen positions to zero and remaining K positions to message bits
            u_sequence = getUSequence(N, K, message, rs)
            encoded_message = encode(u_sequence, n, N)
            
            # Performing BPSK and AWGN
            bpsk_modulated_sequence = BPSK(encoded_message)
            received_sequqnce = AWGN(bpsk_modulated_sequence, Sigma)
            Likelihood = received_sequqnce
            enc_msg_while_decoding, decoded_u_sequence_sc = np.array(decode_sc(Likelihood, 0, n, isFrozen))
            decoded_u_sequence_scl = decode_scl(Likelihood, 0, n, M, isFrozen)[0][1]

            success_flag_sc = True
            success_flag_scl = True
            for i in range(N):
                if u_sequence[i] != decoded_u_sequence_sc[i]:
                    N_bit_errors_sc += 1
                    success_flag_sc = False
                if u_sequence[i] != decoded_u_sequence_scl[i]:
                    N_bit_errors_scl += 1
                    success_flag_scl = False
            if success_flag_sc:
                Success_sc += 1
            if success_flag_scl:
                Success_scl += 1

            # End for
        # End Nsim
        P_success_sc.append(Success_sc / N_sim)
        P_success_scl.append(Success_scl / N_sim)

        BER_simulated_sc.append(N_bit_errors_sc / (N * N_sim))
        BER_simulated_scl.append(N_bit_errors_scl / (N * N_sim))

        BER_Theoretical.append(0.5 * erfc(np.sqrt(Rate * EbNo)))

    plt.figure(1)
    plt.plot(EbNodB_values, np.array(BER_simulated_sc), label = f"N={N}, K={K} Sim BER SC") 
    plt.plot(EbNodB_values, np.array(BER_simulated_scl), label = f"N={N}, K={K} Sim BER SCL") 
    plt.plot(EbNodB_values, BER_Theoretical, label = f"N={N}, K={K} Theory BER", linestyle='--') 
    plt.ylim(1e-4, 1)
    # plt.xlim(0, 5)
    plt.yscale("log")
    plt.xlabel("Eb/No (dB)")  
    plt.ylabel("BER")  
    plt.title("BER v/s Eb/No SC vs SCL")
    plt.grid(True, which="both", axis="both")
    plt.legend()

    plt.figure(2)
    plt.plot(EbNodB_values, np.array(P_success_sc), label = f"N={N}, K={K} Psuccess SC") 
    plt.plot(EbNodB_values, np.array(P_success_scl), label = f"N={N}, K={K} Psuccess SCL") 
    plt.xlabel("Eb/No (dB)")  
    plt.ylabel("P_Success")  
    plt.title("P_Success v/s Eb/No SC vs SCL")
    plt.grid(True)
    plt.legend()

    plt.show()
# N: Total number of bits
# K: Number of message bits
sc_vs_scl(64, 40, 4)
