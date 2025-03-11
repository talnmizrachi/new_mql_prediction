

def create_success_dict(_df):
    only_requested = _df[['mql_date', 'requested_bg']].copy()
    
    only_requested['mql_month'] = only_requested['mql_date'].dt.month
    pivot_ = only_requested[['mql_month', 'requested_bg']].pivot_table(index="mql_month", columns='requested_bg',
                                                                       aggfunc=len, fill_value=0, margins=True)
    
    month_proportion_dict = (pivot_['requested'] / pivot_['All']).to_dict()
    
    return month_proportion_dict