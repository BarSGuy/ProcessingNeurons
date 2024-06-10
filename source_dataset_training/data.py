
import torch_geometric.data as data
import torch_geometric as pyg
import logging
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloader(cfg, batch_size=None):
    if batch_size is not None:
        if cfg.source_dataset.name == "zinc12k":
            return get_zinc12k_dataloader(batch_size=batch_size)
        elif cfg.source_dataset.name == "cifar10":
            return get_cifar10_dataloader(batch_size=batch_size)
        else:
            raise NotImplementedError(f"Source dataset {cfg.source_dataset.name} not implemented")
    else:
        if cfg.source_dataset.name == "zinc12k":
            return get_zinc12k_dataloader(batch_size=128)
        else:
            raise NotImplementedError(f"Source dataset {cfg.source_dataset.name} not implemented")


def get_zinc12k_dataloader(batch_size=128):
    zinc_dataloader = {
        name: data.DataLoader(
            pyg.datasets.ZINC(
                split=name,
                subset=True,
                root='./datasets/zinc',
            ),
            batch_size=batch_size,
            num_workers=4,
            shuffle=(name == "train"),
        )
        for name in ["train", "val", "test"]
    }
    num_elements_in_target = 1
    return zinc_dataloader, num_elements_in_target


def get_cifar10_dataloader(batch_size=128):
    # Define the transform to apply to the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    ])

    # Create a dictionary of DataLoader for train, validation, and test sets
    cifar10_dataloader = {
        'test': DataLoader(
            torchvision.datasets.CIFAR10(
                root='./datasets/cifar10',
                train=False,
                download=True,
                transform=transform
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
    }
    num_elements_in_target = 10
    return cifar10_dataloader, num_elements_in_target

def get_n_max_of_dataset(cfg):
    if cfg.source_dataset.name == "zinc12k":
        try:
            # Load the ZINC12k dataset for all splits
            zinc_dataset = {
                name: pyg.datasets.ZINC(
                    split=name, subset=True, root='./datasets/zinc')
                for name in ["train", "val", "test"]
            }

            # Combine all datasets and find the maximum number of nodes
            max_nodes = max(
                max(data.num_nodes for data in zinc_dataset["train"]),
                max(data.num_nodes for data in zinc_dataset["val"]),
                max(data.num_nodes for data in zinc_dataset["test"])
            )

            logging.info(
                f"Maximum number of nodes in ZINC12k dataset: {max_nodes}")
            cfg.source_dataset.max_nodes = max_nodes
            return max_nodes

        except Exception as e:
            logging.error(
                f"An error occurred while processing the ZINC12k dataset: {e}")
            raise e
    elif cfg.source_dataset.name == "cifar10":
        cfg.source_dataset.max_nodes = -1
        max_nodes = -1
        return max_nodes
    else:
        raise NotImplementedError(
            f"Source dataset {cfg.source_dataset.name} not implemented")


##############################################################
###################### Main architecure ######################
##############################################################

indices_zink12k = torch.tensor([423, 381, 186, 519,  47, 358, 772, 531, 356, 933, 362,  27, 712, 716,
                          713,  96, 151, 570, 274, 798, 667, 242, 526,  77, 120,   8, 212, 393,
                          307, 542, 360, 912, 834, 646, 310, 157, 294, 810, 257, 439, 596,  29,
                          141, 584, 901, 864, 444, 122, 159, 792, 731, 124, 303, 552, 609, 664,
                          545, 377, 149, 399, 129, 179, 189, 918, 243, 654, 895, 638, 199, 279,
                          61, 109, 659, 819,  60, 988, 298, 928, 858, 173,  48, 605, 240, 812,
                          516, 919, 703, 389, 831, 603, 641, 314, 215,  32, 630, 791, 766, 604,
                          839, 255, 275, 931, 777, 789, 806, 688, 752, 827, 701, 253, 945, 282,
                          150, 546, 463, 474, 187, 207, 121, 787, 742, 144, 234, 617, 876, 367,
                          598, 490, 300, 502, 868, 613,  83, 927, 220, 666, 346, 218, 422, 886,
                          847, 822, 751, 844, 651, 459, 296, 633, 996, 551, 863,  26, 973,  54,
                          372, 481, 722, 195, 208, 529, 504, 171,  69, 226, 875, 113,  44, 993,
                          169, 981, 696,  46,   3,  92, 409, 962, 517, 579, 969, 388, 410, 653,
                          34, 108, 835, 237, 140, 369, 123, 191, 938, 645, 293, 332, 893, 665,
                          305, 395, 115, 845, 623, 771, 694, 755, 365, 758,  79,   2, 311, 343,
                          602,  95, 158, 407, 185, 269, 870, 832, 182,  71, 443, 183, 674, 989,
                          94, 497, 125,  18, 112, 278,  88, 139, 271, 192, 745, 355, 263, 732,
                          142, 201, 705, 336,  28, 715, 746, 506, 600, 869, 891, 478, 836, 114,
                          677,  53, 576, 455, 155, 660, 614, 977, 573, 626, 862, 648, 781, 572,
                          560, 402,  63, 649, 998, 267, 914, 266, 612, 710, 374, 761, 768, 511,
                          644, 734, 759, 843, 328, 884,  52, 351, 738,  59, 256, 911, 724, 163,
                          42, 378,  40, 223,  39, 815, 736, 503, 316, 460, 558, 313, 642, 404,
                          130, 903, 882, 203,  86, 350, 485, 930, 441, 593, 799, 238, 524, 371,
                          22, 379, 888, 162, 292, 433, 580, 592, 368, 802, 513, 599, 164, 672,
                          118, 621, 209, 855, 636, 959, 306, 797, 838, 821, 510, 916, 408, 468,
                          193, 753, 309, 451, 322, 574, 848, 907, 486, 461, 776, 606, 774, 265,
                          936,  70, 894, 717, 907, 486, 461, 776, 606, 774, 265, 222, 785,  11,
                          133, 170, 926, 167, 178, 500, 250, 563, 430, 813, 635, 615, 210, 749,
                          675, 947, 691, 111, 955, 946, 807, 905, 401, 908, 442, 764,  45, 213,
                          62, 690, 288, 922, 301, 219, 312, 788, 897, 420,  43, 132,  97,  25,
                          699, 432, 639, 856, 412, 814, 417, 625, 338, 719, 700, 522, 160, 978,
                          786, 967, 386, 934,  38, 249, 673, 197, 727,  31, 100, 920, 890, 881,
                          13, 438,  56, 601, 382, 723, 341,  35, 143,  80, 434,  51, 427, 229,
                          287, 445, 871, 168, 951, 340, 929, 817, 961, 376, 370, 991, 894, 146,
                          533, 778, 695, 406, 373, 971, 145,  16, 347, 949, 686, 663, 403, 308,
                          960, 227, 852, 562, 737, 624,   6, 805, 214, 128, 770, 707, 775, 709,
                          585, 569, 172, 515, 454, 980, 823, 359, 342, 656, 872,  68, 530, 106,
                          910,  17, 939, 297, 231, 495, 841, 472, 597, 453, 935, 259, 661, 733,
                          81, 539, 366, 241, 833, 640, 137,  20, 767, 200, 909, 697, 247, 469,
                          67, 684, 540, 873, 391, 494, 272, 429, 299, 874, 277, 721, 608, 620,
                          78,  93, 435, 543,  91, 689, 575, 477, 107,  55, 629, 941, 826, 493,
                          315, 750, 984, 809, 390, 619, 221, 992, 590, 501, 554,  33, 898, 475,
                          940, 631, 714, 837, 281, 470, 405,  19, 204, 523, 337, 669, 762, 304,
                          0, 854, 528, 643,  74,  15, 757, 937, 329, 594, 610, 578, 966, 254,
                          687, 842, 925, 885,  73, 462, 385, 480, 458, 704, 668, 239,  57, 548,
                          632, 447, 825,  90,  50, 702, 728, 741,   5, 110, 986, 283, 136, 565,
                          248, 720, 782, 261, 174, 188, 156, 784, 119, 780, 102, 577, 830, 567,
                          883,  75, 564, 561,  76, 228, 126, 829, 491,  89, 754, 957, 205, 730,
                          246, 333,  64, 559, 436, 995, 357, 317, 878, 953, 566, 251, 557, 465,
                          166, 800, 344, 380,  72])
