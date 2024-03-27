import numpy as np
from lp_solver import LPProblem, LPSolver


def restricted_master_problem(patterns, demand_nums, prob_type):
    obj_cof = np.ones(patterns.shape[1])
    obj_cof = np.concatenate((obj_cof, np.array([-1])))

    mat_cof = np.column_stack((patterns, np.ones(patterns.shape[0]), demand_nums))
    lbs = np.zeros(patterns.shape[1])
    ub = np.sum(demand_nums)
    ubs = np.full(patterns.shape[1], ub)

    prob = LPProblem(obj_cof, 0, mat_cof, lbs, ubs, prob_type=prob_type)
    sol = LPSolver(prob)
    sol.solve()

    if prob_type == 'lp':
        return np.array(sol.dual_val)
    else:
        return sol.obj, np.array(sol.solution)


def sub_problem(dual_vals, cj, demand_width, width_limit):
    obj_cof = np.concatenate((-1 * dual_vals, np.array([-1])))

    # 列生成规则: 每种pattern产生的长度和小于等于钢管长度
    mat_cof = np.concatenate((demand_width, np.array([-1]), np.array([width_limit])), dtype=int)
    mat_cof = np.reshape(mat_cof, (1, mat_cof.shape[0]))

    lbs = np.zeros(demand_width.shape[0], dtype=int)
    ub = np.floor(width_limit / np.min(demand_width))
    ubs = np.full(demand_width.shape[0], ub)

    prob = LPProblem(obj_cof, cj, mat_cof, lbs, ubs, prob_type='ip')
    sol = LPSolver(prob)
    sol.solve()

    if sol.obj >= 0:
        return False, np.array([])
    return True, np.array(sol.solution)


roll_width = np.array(17)
demand_width_array = np.array([3, 6, 7, 8], dtype=int)
demand_number_array = np.array([25, 20, 18, 10], dtype=int)
init_patterns = np.diag(np.floor(roll_width / demand_width_array))

has_new_pattern = True
new_pattern = None
cur_patterns = init_patterns
while has_new_pattern:
    if new_pattern is not None:
        cur_patterns = np.column_stack((cur_patterns, new_pattern))
    dual_values = restricted_master_problem(cur_patterns, demand_number_array, 'lp')
    has_new_pattern, new_pattern = sub_problem(dual_values, 1, demand_width_array, roll_width)
    print()

obj, solution = restricted_master_problem(cur_patterns, demand_number_array, 'ip')
print()
print('====' * 5)
print(f'obj: {obj}')
print(f'sol: {solution}')
print(f'patterns: ')
print(cur_patterns)
