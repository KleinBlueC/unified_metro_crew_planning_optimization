def get_power_set(orig_list: list) -> list:
    """
    orig_list: a list, where each element is a qualification list of lines
    return a list of list
    """
    if len(orig_list) == 0:
        return []
    elem = orig_list.pop()
    ret = get_power_set(orig_list)
    for p_elem in ret[:]: # ret changes during the for iteration, so use a copy to avoid unexpected behaviors
        ret.append([elem] + p_elem)
    ret.append([elem])
    return sorted(ret, key=lambda x:len(x))

def get_union_supeset_collections(clist:list, all_q:list) -> list:
    """
    clist: the list to be performed, where each element is a qualification list of lines
    all_q: the whole search space for supersets
    """
    ret = []
    for qs in clist:
        for q in qs[:]:
            for q_ref in all_q:
                if all(l in q_ref for l in q) and len(q_ref) > len(q) and q_ref not in qs: # is superset and to be added
                    qs.append(q_ref)
        ret.append(sorted(qs, key=lambda x:len(x)))
    return ret

### 20241120 ReWrite
def get_superset_group(q, Q) -> list:
    """
    return a list of qualifications that contains all supersets of q
    """
    Eq = [q]
    for q_ref in Q:
        if all(l in q_ref for l in q) and len(q_ref) > len(q) and q_ref not in Eq: # is superset and to be added
            Eq.append(q_ref)
    return tuple(sorted(Eq, key=lambda x:len(x)))


    


if __name__ == "__main__":
    print(get_power_set([[11, 22], [22], [333], [0]]))
    print(get_union_supeset_collections([[[1]], [[2]],[[3]], [[1], [2]], [[1, 2]], [[1, 3]], [[1, 2], [2, 3]]], [[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]))

        
    