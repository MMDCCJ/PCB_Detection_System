def pack_pcb_list(p_list:list):
    """
        打包PCB_List
    """
    pcb_list = []
    for i in range(len(p_list)):
        pcb = {}
        pcb['pcb_id'] = p_list[i][0]
        pcb['user_id'] = p_list[i][1]
        pcb['image_name'] = p_list[i][2]
        pcb['update_date'] = p_list[i][3].strftime('%Y-%m-%d %H:%M:%S')
        pcb_list.append(pcb)
    return pcb_list