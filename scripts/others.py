import datetime

def elapsed_time(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        start_time.strftime("%H:%M:%S")
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        print("===" * 15)
        print("Elapsed time: ", end_time - start_time)
        print("===" * 15)
        return result
    return wrapper

def ask_boost_round():
    ask_boost_round = input("Do you want to change the number of boosting rounds? [100]: ")
    if ask_boost_round:
        try:
            num_boost_round = int(ask_boost_round)
            print(f"Number of boosting rounds: {num_boost_round}")
        except ValueError:
            raise ValueError("Please enter an integer value.")
    else:
        num_boost_round = 100
        print(f"Number of boosting rounds: {num_boost_round}")
    return num_boost_round

def ask_n_trials():
    ask_n_trials = input("Do you want to change the number of n_trials? [50]: ")
    if ask_n_trials:
        try:
            N_TRIALS = int(ask_n_trials)
            print(f"Number of trials: {N_TRIALS}")
        except ValueError:
            raise ValueError("Please enter an integer value.")
    else:
        N_TRIALS = 50
        print(f"Number of trials: {N_TRIALS}")
    return N_TRIALS