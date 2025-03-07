
# %%
"""© 2025 Meret Preuß <meret.preuss@uol.de>"""
import scipy as sp
class AbstractSeries:
    """Generate the abstract Kato perturbational series up to a given order.

    This class provides methods to generate the perturbational power series 
    as specified by Tosio Kato, which is a series of matrix elements
    used in perturbation theory calculations.
    The series is generated up to a specified maximum order.
    See the attached documentation for more information on the physics behind this series.

    Attributes:
        max_order (int): The maximum order of the series.
        return_without_first_order (bool): Flag indicating whether to return 
                                            the series without the first order.
                                            This is useful if the first order is 
                                            analytically known to vanish.

    Methods:
        __init__(max_order: int, return_without_first_order: bool = False) -> None:
            Initializes the AbstractSeries object with the specified maximum order and return flag.

        _configuration(init_pos: int, init_sum: int, rem_pos: int, 
                                    rem_sum: int, config: list) -> None:
            Solves a needed combinatorial problem:
            Generates all possible configurations of init_sum objects onto 
            int_pos positions (recursively).

        _outer_zeros(order: int) -> None:
            Generates all contributions in standard form where the outer k's are both 0.

        _outer_nonzeros(order: int) -> None:
            Generates all contributions where both k_L and k_R are non-zero for one fixed order.

        _generate_single_order_into_cache(current_order: int) -> None:
            Generates all DMEs of a fixed order and stores them in the cache.

        _factor_into_emes(dme: list) -> list:
            Splits a list of integers at the integers that are 0.

        _label_eme(eme: list) -> int:
            Returns the minimal Z and the respective minimal eme, and adds the calculated
            values to the global dictionary of eme labels.

        _generate_all_orders() -> None:
            Generates the whole (abstract, i.e. symbolic) Kato series up to the maximum order.

        _create_eme_occupation_dict(factored_matrixelement: list) -> dict:
            Creates a (sorted) dictionary with the occupation of each EME within a DME.

        print_abstract_result(result_dict: dict) -> None:
            Prints the abstract Kato series.

        return_all_orders(print_results: bool = False) -> dict:
            Returns the full series of abstract Kato series up
            to the maximum order in a dictionary.
    """

    def __init__(
        self, max_order: int, return_without_first_order: bool = False
    ) -> None:
        self.max_order = max_order
        # 1. Parameters for the generation of a single order
        self.first_call = True
        self.list_idx = 0
        self.length_cache = [0]
        self.config_cache = []
        self.direct_dmes = []
        self.indirect_dmes = []  # position in list equals 2+eta
        self.weights_indirect_dmes = {}
        self.return_without_first_order = return_without_first_order

        # 2. Parameters for the generation of the full series

        self.dme_all_orders = {}

        for i in range(self.max_order + 1):
            # value should be a dictionary of dictionaries.
            # each dictionary represents a DME.
            # For this: keys: 'weight', 'EMEs', 'factors', 'repres'
            # 'EME' itself is a dict with keys: 'label' and 'occupation'
            self.dme_all_orders[i] = {}

            # list for already treated matrix elements, identification by EME occupation
            # ! list has empty zeroeth entry, so the order indexing is more readable
            self.treated_dmes = [{} for _ in range(self.max_order + 1)]
            # initialize dict for EME labels
            self.dict_eme_labels = {}

    def _configuration(
        self,
        init_pos: int,
        init_sum: int,
        rem_pos: int,
        rem_sum: int,
        config: list,
    ) -> None:
        """Generate all possible configurations of init_sum objects onto 
        init_pos positions"""
        # if the function is called for the first time, reset the current_list track_keeper
        if self.first_call:
            # create list with enough entries to store all possible configurations
            self.length_cache = int(sp.special.binom(init_sum + init_pos - 1, init_sum))
            self.config_cache = [[] for _ in range(self.length_cache)]
            self.first_call = False
            config = []

        if rem_pos == 1:
            self.config_cache[self.list_idx] = config + [rem_sum]
            self.list_idx += 1
        else:
            for i in range(rem_sum + 1):
                self._configuration(
                    init_pos,
                    init_sum,
                    rem_pos - 1,
                    rem_sum - i,
                    config + [i],
                )

        if self.list_idx == self.length_cache:
            self.list_idx = 0
            self.first_call = True

    def _outer_zeros(self, order: int) -> None:
        """Generate all contribution in standardform where the outer k's are both 0"""
        self.config_cache = []
        self._configuration(order - 1, order - 1, order - 1, order - 1, [])
        self.direct_dmes = self.config_cache

    def _outer_nonzeros(self, order: int) -> None:
        """Generate all contributions where both k_L and k_R are non-zero for 
        one fixed order"""
        self.indirect_dmes = (
            []
        )  # becomes a list (eta values) of lists (of fixed-eta configurations)
        self.weights_indirect_dmes = {}
        self.config_cache = []
        eta_range = range(2, order)  # eta can take all values from 2 to order-1
        for eta in eta_range:
            sum_ = order - 1 - eta
            pos = order - 1
            self._configuration(pos, sum_, pos, sum_, [])
            self.indirect_dmes.append(self.config_cache)

        # write contributions in standardform (without explicit outer zeros).
        # For this, look at each contribution
        for eta_idx in range(len(self.indirect_dmes)):
            eta = eta_idx + 2  # eta ranges from 2 to order-1
            for contr_idx in range(len(self.indirect_dmes[eta_idx])):
                k_list = self.indirect_dmes[eta_idx][contr_idx]
                idx = 0
                not_found = True
                while not_found:
                    not_found = k_list[idx] != 0
                    idx += 1
                # choose first zero, write down part after it,
                # then eta, then rest of the list without
                # (0 is used for DME form)
                idx -= 1  # this way, idx is the index of zero
                new_list = k_list[idx + 1 :]
                new_list.append(eta)
                if idx != 0:
                    new_list += k_list[:idx]
                self.indirect_dmes[eta_idx][contr_idx] = new_list
                self.weights_indirect_dmes[tuple(new_list)] = -(eta - 1)

    def _generate_single_order_into_cache(self, current_order: int) -> None:
        """Generate all DMEs of a fixed order and stores them in the cache"""
        if current_order == 1:
            self.direct_dmes = [[0]]
            self.indirect_dmes = []
            self.weights_indirect_dmes = {}
        else:
            self._outer_zeros(current_order)
            self._outer_nonzeros(current_order)

    def _factor_into_emes(self, dme: list) -> list:
        """Split a list of integers at the integers that are 0"""
        if dme == [0]:
            return [[]]
        new_list = []
        last_zero_idx = 0
        for i in range(len(dme)):
            if dme[i] == 0:
                new_list = new_list + [dme[last_zero_idx:i]]
                last_zero_idx = i + 1  # this way, 0 is not included in the next slice
        new_list = new_list + [dme[last_zero_idx:]]
        return new_list

    def _label_eme(self, eme: list) -> int:
        """Return the minimal Z and the respective minimal eme,
        adds the calculated values to the global dictionary of eme labels
        """
        Z = 0
        Z_ = 0
        eme_rev = eme[::-1]
        # loop is not executed if len(eme)==0, i.e. if eme = <a|V|a>
        for i in range(len(eme)):
            Z += eme[i] * pow(self.max_order, i)
            Z_ += eme_rev[i] * pow(self.max_order, i)

        if Z > Z_:
            label_min = Z_
            eme_min = eme_rev
        else:
            label_min = Z
            eme_min = eme
        self.dict_eme_labels[label_min] = eme_min
        return label_min

    def _generate_all_orders(self) -> None:
        "Generate whole (abstract, i.e. symbolic) Kato series up to max_order"
        for n in range(1, self.max_order + 1):
            print(f"Generating order n= {n}...")
            self._generate_single_order_into_cache(n)
            # contributions for order n now lie in self.direct_dmes and self.indirect_dmes
            # Go through each direct DME and add weight from indirect DMEs
            # Then, factor each DME into EMEs and check if a matrix element
            # with the same EMEs has already been treated.
            # In this case: combine the weights. If not create entry in 
            # self.treated_matrix_elements
            # and append result to self.dme_all_orders[o]

            for dme_tuple in self.direct_dmes:
                factored = self._factor_into_emes(dme_tuple)
                # each split produces a minus sign
                sign_matrix_element = pow(-1, len(factored) - 1)
                weight_nonzero = self.weights_indirect_dmes.get(tuple(dme_tuple), 0)
                weight = weight_nonzero + 1
                weight = weight * sign_matrix_element
                emes = self._create_eme_occupation_dict(factored)
                if str(emes) in self.treated_dmes[n]:
                    key = self.treated_dmes[n][str(emes)]
                    self.dme_all_orders[n][key]["weight"] += weight
                else:
                    dict_dme = {
                        "weight": weight,
                        "EMEs": emes,
                        "factors": factored,
                        "repres": dme_tuple,
                    }
                    key_in_dict = (
                        max(self.dme_all_orders[n].keys()) + 1
                        if self.dme_all_orders[n]
                        else 0
                    )
                    self.dme_all_orders[n][key_in_dict] = dict_dme
                    self.treated_dmes[n][str(emes)] = key_in_dict
            print("Generation of abstract Kato series complete.")

    def _create_eme_occupation_dict(self, factored_matrixelement: list) -> dict:
        """Create a (sorted) dictionary with the occupation of each EME within a DME"""
        occ_dict = {}
        # itterate through all EMEs within a matrix element
        for i in range(len(factored_matrixelement)):
            label_min = self._label_eme(factored_matrixelement[i])
            occ_dict[label_min] = occ_dict.get(label_min, 0) + 1
            # all dicts should have same order
        return dict(sorted(occ_dict.items()))

    def print_abstract_result(self, result_dict: dict) -> None:
        """Print the abstract Kato series"""
        print(f"\n Results:\n")
        for n in range(1, self.max_order + 1):
            print(f"Order n= {n}")
            for i in range(max(result_dict["full series"][n].keys()) + 1):
                print(result_dict["full series"][n][i])
            print("\n")

    def return_all_orders(self, print_results: bool = False) -> dict:
        """Return the full series of abstract Kato series up to max_order in a dictionary"""
        self._generate_all_orders()
        self.dme_all_orders.pop(0)
        contributions = self.dme_all_orders.copy()

        if self.return_without_first_order:
            no_first_order = {}
            for n in range(1, self.max_order + 1):
                no_first_order[n] = {
                    key: value
                    for key, value in contributions[n].items()
                    if not value.get("EMEs").get(0)
                }
                # Here, only entries that do not contain the first order correction are kept
            return_dict = {
                "full series": contributions,
                "no first order": no_first_order,
                "eme labels": self.dict_eme_labels.copy(),
            }
        else:
            return_dict = {
                "full series": contributions,
                "eme labels": self.dict_eme_labels.copy(),
            }
        if print_results:
            self.print_abstract_result(return_dict)
        return return_dict


if __name__ == "__main__":
    series = AbstractSeries(9)
    results = series.return_all_orders(print_results=True)
