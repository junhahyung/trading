# standard expectunity formula
def expectancy(wins, losses):

    try: 
        AW = sum(wins)/len(wins)
        PW = len(wins)/(len(wins) + len(losses))
 
        AL = sum(losses)/len(losses)
        PL = len(losses)/(len(wins) + len(losses))

        return (AW * PW + AL * PL)/abs(AL)
    
    except ZeroDivisionError:
        return 0

def expectunity(wins, losses, strat_cal_days):

    try:
        # Calculate the opportunity value
        num_trades = len(wins) + len(losses)
        #strat_cal_days = len(trans)
        strat_cal_days= max(strat_cal_days,1)  # avoid divide by zero
        opportunities = num_trades * 520/strat_cal_days
    
        return expectancy(wins, losses) * opportunities
    
    except ZeroDivisionError:
        return 0  
