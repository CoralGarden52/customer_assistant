# 未映射图片排查报告

## 总览
- 插图目录文件数：2608
- 文本中唯一图片引用数：2569
- 已映射唯一图片数：2271
- 文本中未映射唯一图片数：298
- 文本未引用的图片文件数：39

## 结论分组
- `源数据异常`：手册文本中的 `<PIC>` 数量与图片列表长度不一致，这类问题会直接造成图片无法正确对齐。
- 可编程温控器手册.txt：`<PIC>=57`，`images=58`，差值 `-1`
- 发电机手册.txt：`<PIC>=102`，`images=101`，差值 `1`
- 洗碗机手册.txt：`<PIC>=29`，`images=0`，差值 `29`
- `索引/文本格式异常`：documents.parquet 中仍是原始 JSON 风格文本的手册。
- 汇总英文手册.txt
- 洗碗机手册.txt
- `定位失败（span_not_found）`：本次 298 张未映射图片主要来自 text_unit 定位失败，集中在少数手册。

## 未映射图片最多的手册
- 发电机手册.txt：未映射 57 / 100，未定位 text_unit 13 个，short_id 区间 898-902, 904-905, 908, 912-914, 916-917，示例 ['Manual18_10', 'Manual18_11', 'Manual18_12', 'Manual18_13', 'Manual18_14', 'Manual18_15', 'Manual18_16', 'Manual18_19']
- 相机手册.txt：未映射 45 / 87，未定位 text_unit 11 个，short_id 区间 950-952, 954-956, 958, 960, 962-963, 971，示例 ['Manual29_1', 'Manual29_10', 'Manual29_11', 'Manual29_12', 'Manual29_13', 'Manual29_14', 'Manual29_15', 'Manual29_16']
- 水泵手册.txt：未映射 28 / 78，未定位 text_unit 5 个，short_id 区间 825, 827-828, 830-831，示例 ['Manual31_10', 'Manual31_11', 'Manual31_12', 'Manual31_13', 'Manual31_14', 'Manual31_23', 'Manual31_24', 'Manual31_25']
- 冰箱手册.txt：未映射 25 / 58，未定位 text_unit 8 个，short_id 区间 833, 835-836, 838, 840-841, 844, 847，示例 ['Manual17_10', 'Manual17_11', 'Manual17_12', 'Manual17_13', 'Manual17_14', 'Manual17_15', 'Manual17_16', 'Manual17_17']
- 空调手册.txt：未映射 22 / 36，未定位 text_unit 8 个，short_id 区间 932, 934, 938-939, 941, 943, 946, 948，示例 ['Manual01_0', 'Manual01_1', 'Manual01_12', 'Manual01_13', 'Manual01_14', 'Manual01_17', 'Manual01_18', 'Manual01_19']
- 健身追踪器手册.txt：未映射 21 / 56，未定位 text_unit 13 个，short_id 区间 56, 58-60, 64, 67, 69, 71, 74-75, 78-80，示例 ['Manual16_0', 'Manual16_10', 'Manual16_26', 'Manual16_29', 'Manual16_30', 'Manual16_33', 'Manual16_38', 'Manual16_39']
- 摩托艇手册.txt：未映射 20 / 35，未定位 text_unit 6 个，short_id 区间 853, 858, 860, 862, 865-866，示例 ['Manual40_0', 'Manual40_10', 'Manual40_11', 'Manual40_12', 'Manual40_16', 'Manual40_17', 'Manual40_18', 'Manual40_21']
- 健身单车手册.txt：未映射 16 / 43，未定位 text_unit 6 个，short_id 区间 89-90, 92, 95-96, 110，示例 ['Manual14_11', 'Manual14_12', 'Manual14_13', 'Manual14_14', 'Manual14_15', 'Manual14_16', 'Manual14_17', 'Manual14_18']
- 可编程温控器手册.txt：未映射 16 / 58，未定位 text_unit 4 个，short_id 区间 180, 183, 185, 188，示例 ['Manual36_21', 'Manual36_22', 'Manual36_23', 'Manual36_24', 'Manual36_25', 'Manual36_26', 'Manual36_32', 'Manual36_33']
- VR头显手册.txt：未映射 9 / 10，未定位 text_unit 5 个，short_id 区间 152, 154, 156-158，示例 ['Manual38_1', 'Manual38_2', 'Manual38_3', 'Manual38_4', 'Manual38_5', 'vr_01', 'vr_02', 'vr_03']
- 吹风机手册.txt：未映射 8 / 54，未定位 text_unit 5 个，short_id 区间 39, 45, 47, 51, 53，示例 ['Manual04_25', 'Manual04_26', 'Manual04_27', 'Manual04_28', 'Manual04_36', 'Manual04_37', 'Manual04_43', 'Manual04_46']
- 烤箱手册.txt：未映射 8 / 30，未定位 text_unit 6 个，short_id 区间 880, 885, 887, 892-893, 895，示例 ['Manual28_6', 'oven_01', 'oven_02', 'oven_03', 'oven_04', 'oven_05', 'oven_06', 'oven_07']
- 蒸汽清洁机手册.txt：未映射 8 / 17，未定位 text_unit 4 个，short_id 区间 869, 871, 873, 875，示例 ['Manual05_10', 'Manual05_11', 'Manual05_12', 'Manual05_3', 'Manual05_4', 'Manual05_5', 'Manual05_8', 'Manual05_9']
- 电钻手册.txt：未映射 5 / 22，未定位 text_unit 12 个，short_id 区间 119, 124-125, 129-132, 135, 137, 141, 144-145，示例 ['Manual11_1', 'Manual11_9', 'drill0_14', 'drill0_17', 'drill0_18']
- 儿童电动摩托车手册.txt：未映射 4 / 23，未定位 text_unit 4 个，short_id 区间 921, 927-929，示例 ['Manual26_11', 'Manual26_12', 'Manual26_13', 'Manual26_17']
- 人体工学椅手册.txt：未映射 2 / 16，未定位 text_unit 1 个，short_id 区间 878，示例 ['Manual02_14', 'Manual02_15']
- 空气净化器手册.txt：未映射 2 / 27，未定位 text_unit 3 个，short_id 区间 112, 116, 118，示例 ['Manual03_20', 'Manual03_26']
- 蓝牙激光鼠标手册.txt：未映射 2 / 19，未定位 text_unit 2 个，short_id 区间 162, 166，示例 ['Manual27_10', 'Manual27_11']

## 文本未引用的图片文件示例
- 共 39 张，示例：['Camera_20', 'Camera_66', 'Camera_67', 'Dish_washer_01', 'Dish_washer_02', 'Dish_washer_03', 'Dish_washer_04', 'Dish_washer_05', 'Dish_washer_06', 'Dish_washer_07', 'Dish_washer_08', 'Manual06_0', 'Manual06_1', 'Manual06_10', 'Manual06_11', 'Manual06_12', 'Manual06_13', 'Manual06_14', 'Manual06_15', 'Manual06_16']

## 判断
- 这 298 张未映射图片的主因不是图片文件缺失；`missing_in_folder = 0`。
- 主因也不是 GraphRAG 完全没读到这些手册，因为对应手册都已有部分图片成功映射。
- 主要问题是少数手册中，连续若干 `text_unit` 无法在源文本中稳定定位，导致这些 `text_unit` 覆盖到的图片没有被挂上。
- 需要优先处理的不是全局重建索引，而是增强这些手册的 `text_unit -> source span` 对齐策略。

## 优先排查建议
- 第一优先：`发电机手册`、`相机手册`、`水泵手册`、`冰箱手册`、`空调手册`。这些手册贡献了最多未映射图片。
- 第二优先：修复 `洗碗机手册` 的源数据，当前有 `<PIC>` 但图片列表为空。
- 第三优先：针对连续 short_id 区间做更强的定位回退，例如缩短匹配窗口、允许跨段前后缀匹配，或基于相邻已定位 text_unit 约束搜索范围。
