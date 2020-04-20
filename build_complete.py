exchange_str = '[mark] [mark] [mark] ，2015年3月7日，[mark] 黄某[mark] [mark] [mark] 、[mark] [mark] [mark] ××[mark] ××[mark] [mark] ××[mark] [mark] ，[mark] [mark] [mark] 。[mark] 15[mark] ，其[mark] 了[mark] 卢某[mark] 在[mark] [mark] [mark] 的[mark] [mark] [mark] [mark] [mark] [mark] （经[mark] ，[mark] [mark] 6670[mark] ），[mark] 黄某随后用[mark] [mark] [mark] [mark] ，用[mark] [mark] [mark] [mark] [mark] 的[mark] ，用[mark] [mark] [mark] 后[mark] [mark] ，[mark] [mark] 至其[mark] [mark] [mark] 的[mark] 内[mark] ，后[mark] 后得[mark] 。2015年3月28日，[mark] 黄某因[mark] 被[mark] [mark] 二年。同年7月15日，[mark] 黄某向[mark] [mark] 。'

word_list = ['公诉', '机关', '指控', '被告人', '携带', '作案工具', '螺丝刀', '老虎钳', '来到', '珠海市', '区', '镇', '开发区', '银行', '附近', '伺机', '盗窃', '摩托车', '当天', '时许', '发现', '被害人', '停放', '农商', '银行', '门口', '一辆', '银色', '本田', '牌', '女装', '摩托车', '鉴定', '价值', '人民币', '元', '被告人', '螺丝刀', '撬开', '车头', '锁', '老虎钳', '剪断', '控制', '摩托车', '发动机', '电线', '脚蹬', '启动', '摩托车', '驾驶', '逃离', '将车', '驾驶', '位于', '白蕉', '开发区', '出租屋', '藏匿', '转售', '赃款', '被告人', '吸毒', '强制', '戒毒', '被告人', '戒毒所', '投案']

def build_complete_sentence(exchange_str, word_list):
    out_str = exchange_str
    for item in word_list:
        out_str = out_str.replace('[mark]', item, 1)
    return out_str

build_complete_sentence(exchange_str, word_list)