/** Copyright (c) 2022-2026  Elias Fernandez
*
* This file is part of EGTtools.
*
* EGTtools is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* EGTtools is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with EGTtools.  If not, see <http://www.gnu.org/licenses/>
*/
#include "behaviors.hpp"

namespace egttools {
    std::string call_get_action(const py::list &strategies, size_t time_step, int prev_action) {
        std::stringstream result;
        result << "(";
        bool first = true;
        for (py::handle strategy: strategies) {
            if (!first) result << ", ";
            result << py::cast<egttools::FinitePopulations::behaviors::AbstractNFGStrategy *>(strategy)
                    ->get_action(time_step, prev_action);
            first = false;
        }
        result << ")";
        return result.str();
    }
} // namespace egttools

void init_behaviors(py::module_ &mBehaviors) {
    auto mCRD = mBehaviors.def_submodule(
        "CRD",
        "Strategies for collective-risk dilemma games."
    );
    auto mNF = mBehaviors.def_submodule(
        "NormalForm",
        "Strategies for normal-form games."
    );
    auto mNFTwoActions = mNF.def_submodule(
        "TwoActions",
        "Strategies for two-action normal-form games."
    );

    mBehaviors.doc() =
            "The `egttools.numerical.behaviors` submodule contains the available strategies to evolve.";
    mNF.doc() =
            "The `egttools.numerical.behaviors.NormalForm` submodule contains the strategies for normal-form games.";
    mCRD.doc() =
            "The `egttools.numerical.behaviors.CRD` submodule contains the strategies for collective-risk dilemma games.";
    mNFTwoActions.doc() =
            "The `egttools.numerical.behaviors.NormalForm.TwoActions` submodule contains the strategies for "
            "two-action normal-form games.";

    mBehaviors.def(
        "call_get_action",
        &egttools::call_get_action,
        R"pbdoc(
            Return a string representation of the actions chosen by a list of strategies.

            Parameters
            ----------
            strategies : list[AbstractNFGStrategy]
                Strategies to query.
            time_step : int
                Current round.
            prev_action : int
                Previous action of the opponent.

            Returns
            -------
            str
                Tuple-like string with the action chosen by each strategy.
        )pbdoc",
        py::arg("strategies"),
        py::arg("time_step"),
        py::arg("prev_action")
    );

    py::class_<egttools::FinitePopulations::behaviors::AbstractNFGStrategy,
                stubs::PyAbstractNFGStrategy>(mNF, "AbstractNFGStrategy",
                                              R"pbdoc(
            Abstract base class for strategies in repeated two-action normal-form games.
        )pbdoc")
            .def(py::init<>())
            .def(
                "get_action",
                &egttools::FinitePopulations::behaviors::AbstractNFGStrategy::get_action,
                R"pbdoc(
                Return the action chosen by the strategy.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    Action selected by the strategy.

                See Also
                --------
                egttools.behaviors.NormalForm.TwoActions.Cooperator
                egttools.behaviors.NormalForm.TwoActions.Defector
                egttools.behaviors.NormalForm.TwoActions.Random
                egttools.behaviors.NormalForm.TwoActions.TFT
                egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT
                egttools.behaviors.NormalForm.TwoActions.GradualTFT
                egttools.behaviors.NormalForm.TwoActions.ImperfectTFT
                egttools.behaviors.NormalForm.TwoActions.TFTT
                egttools.behaviors.NormalForm.TwoActions.TTFT
                egttools.behaviors.NormalForm.TwoActions.GRIM
                egttools.behaviors.NormalForm.TwoActions.Pavlov
            )pbdoc",
                py::arg("time_step"),
                py::arg("action_prev")
            )
            .def(
                "type",
                &egttools::FinitePopulations::behaviors::AbstractNFGStrategy::type,
                "Return the strategy type."
            )
            .def(
                "is_stochastic",
                &egttools::FinitePopulations::behaviors::AbstractNFGStrategy::is_stochastic,
                "Return whether the strategy is stochastic."
            );

    py::class_<egttools::FinitePopulations::behaviors::AbstractCRDStrategy,
                stubs::PyAbstractCRDStrategy>(mCRD, "AbstractCRDStrategy",
                                              R"pbdoc(
            Abstract base class for strategies in collective-risk dilemma games.
        )pbdoc")
            .def(py::init<>())
            .def(
                "get_action",
                &egttools::FinitePopulations::behaviors::AbstractCRDStrategy::get_action,
                R"pbdoc(
                Return the action chosen by the strategy.

                Parameters
                ----------
                time_step : int
                    Current round.
                group_contributions_prev : int
                    Sum of contributions of the other group members in the previous round.

                Returns
                -------
                int
                    Action selected by the strategy.

                See Also
                --------
                egttools.behaviors.CRD.CRDMemoryOnePlayer
            )pbdoc",
                py::arg("time_step"),
                py::arg("group_contributions_prev")
            )
            .def(
                "type",
                &egttools::FinitePopulations::behaviors::AbstractCRDStrategy::type,
                "Return the strategy type."
            );

    py::class_<egttools::FinitePopulations::behaviors::twoActions::Cooperator,
                egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(
                mNFTwoActions, "Cooperator",
                "Strategy that always cooperates."
            )
            .def(py::init<>())
            .def(
                "get_action",
                &egttools::FinitePopulations::behaviors::twoActions::Cooperator::get_action,
                R"pbdoc(
                Return the action chosen by the strategy.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    Action selected by the strategy.
            )pbdoc",
                py::arg("time_step"),
                py::arg("action_prev")
            )
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::Cooperator::type,
                 "Return the strategy type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::Cooperator::is_stochastic,
                 "Return whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::Defector,
                egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(
                mNFTwoActions, "Defector",
                "Strategy that always defects."
            )
            .def(py::init<>())
            .def(
                "get_action",
                &egttools::FinitePopulations::behaviors::twoActions::Defector::get_action,
                R"pbdoc(
                Return the action chosen by the strategy.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    Action selected by the strategy.
            )pbdoc",
                py::arg("time_step"),
                py::arg("action_prev")
            )
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::Defector::type,
                 "Return the strategy type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::Defector::is_stochastic,
                 "Return whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::RandomPlayer,
                egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(
                mNFTwoActions, "Random",
                "Strategy that cooperates with probability 0.5."
            )
            .def(py::init<>())
            .def(
                "get_action",
                &egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::get_action,
                R"pbdoc(
                Return the action chosen by the strategy.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    Action selected by the strategy.
            )pbdoc",
                py::arg("time_step"),
                py::arg("action_prev")
            )
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::type,
                 "Return the strategy type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::is_stochastic,
                 "Return whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::TitForTat,
                egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(
                mNFTwoActions, "TFT",
                "Tit for Tat: cooperate in the first round, then copy the opponent's previous action."
            )
            .def(py::init<>())
            .def(
                "get_action",
                &egttools::FinitePopulations::behaviors::twoActions::TitForTat::get_action,
                R"pbdoc(
                Return the action chosen by the strategy.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    Action selected by the strategy.
            )pbdoc",
                py::arg("time_step"),
                py::arg("action_prev")
            )
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::TitForTat::type,
                 "Return the strategy type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::TitForTat::is_stochastic,
                 "Return whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT,
                egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(
                mNFTwoActions, "SuspiciousTFT",
                "Suspicious Tit for Tat: defect in the first round, then copy the opponent's previous action."
            )
            .def(py::init<>())
            .def(
                "get_action",
                &egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::get_action,
                R"pbdoc(
                Return the action chosen by the strategy.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    Action selected by the strategy.
            )pbdoc",
                py::arg("time_step"),
                py::arg("action_prev")
            )
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::type,
                 "Return the strategy type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::is_stochastic,
                 "Return whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::GenerousTFT,
                egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(
                mNFTwoActions, "GenerousTFT",
                "Generous Tit for Tat."
            )
            .def(
                py::init<double, double, double, double>(),
                R"pbdoc(
                Construct a Generous Tit for Tat strategy.

                Following a defection, the strategy cooperates with probability

                p(R, P, T, S) = min(1 - (T - R) / (R - S), (R - P) / (T - P))

                where R, P, T, and S are the reward, punishment, temptation, and sucker's payoff.

                Parameters
                ----------
                R : float
                    Reward payoff.
                P : float
                    Punishment payoff.
                T : float
                    Temptation payoff.
                S : float
                    Sucker's payoff.
            )pbdoc",
                py::arg("R"),
                py::arg("P"),
                py::arg("T"),
                py::arg("S")
            )
            .def(
                "get_action",
                &egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::get_action,
                R"pbdoc(
                Return the action chosen by the strategy.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    Action selected by the strategy.
            )pbdoc",
                py::arg("time_step"),
                py::arg("action_prev")
            )
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::type,
                 "Return the strategy type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::is_stochastic,
                 "Return whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::GradualTFT,
                egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(
                mNFTwoActions, "GradualTFT",
                "Gradual Tit for Tat."
            )
            .def(
                py::init<>(),
                R"pbdoc(
                Tit for Tat with two modifications:

                1. The number of punishing defections increases with each additional defection by the opponent.
                2. After each punishment phase, the strategy apologizes by cooperating in the following two rounds.
            )pbdoc"
            )
            .def(
                "get_action",
                &egttools::FinitePopulations::behaviors::twoActions::GradualTFT::get_action,
                R"pbdoc(
                Return the action chosen by the strategy.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    Action selected by the strategy.
            )pbdoc",
                py::arg("time_step"),
                py::arg("action_prev")
            )
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::GradualTFT::type,
                 "Return the strategy type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::GradualTFT::is_stochastic,
                 "Return whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT,
                egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(
                mNFTwoActions, "ImperfectTFT",
                "Tit for Tat with implementation errors."
            )
            .def(
                py::init<double>(),
                R"pbdoc(
                Construct an Imperfect Tit for Tat strategy.

                Parameters
                ----------
                error_probability : float
                    Probability of choosing the opposite action by mistake.
            )pbdoc",
                py::arg("error_probability")
            )
            .def(
                "get_action",
                &egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::get_action,
                R"pbdoc(
                Return the action chosen by the strategy.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    Action selected by the strategy.
            )pbdoc",
                py::arg("time_step"),
                py::arg("action_prev")
            )
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::type,
                 "Return the strategy type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::is_stochastic,
                 "Return whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::TFTT,
                egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(
                mNFTwoActions, "TFTT",
                "Tit for two tats: defect only after two consecutive defections."
            )
            .def(py::init<>())
            .def(
                "get_action",
                &egttools::FinitePopulations::behaviors::twoActions::TFTT::get_action,
                R"pbdoc(
                Return the action chosen by the strategy.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    Action selected by the strategy.
            )pbdoc",
                py::arg("time_step"),
                py::arg("action_prev")
            )
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::TFTT::type,
                 "Return the strategy type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::TFTT::is_stochastic,
                 "Return whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::TTFT,
                egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(
                mNFTwoActions, "TTFT",
                "Two tits for tat: defect twice after a defection."
            )
            .def(py::init<>())
            .def(
                "get_action",
                &egttools::FinitePopulations::behaviors::twoActions::TTFT::get_action,
                R"pbdoc(
                Return the action chosen by the strategy.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    Action selected by the strategy.
            )pbdoc",
                py::arg("time_step"),
                py::arg("action_prev")
            )
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::TTFT::type,
                 "Return the strategy type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::TTFT::is_stochastic,
                 "Return whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::GRIM,
                egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(
                mNFTwoActions, "GRIM",
                "Grim Trigger: cooperate until the opponent defects once, then defect forever."
            )
            .def(py::init<>())
            .def(
                "get_action",
                &egttools::FinitePopulations::behaviors::twoActions::GRIM::get_action,
                R"pbdoc(
                Return the action chosen by the strategy.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    Action selected by the strategy.
            )pbdoc",
                py::arg("time_step"),
                py::arg("action_prev")
            )
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::GRIM::type,
                 "Return the strategy type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::GRIM::is_stochastic,
                 "Return whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::Pavlov,
                egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(
                mNFTwoActions, "Pavlov",
                "Win-stay, lose-shift strategy."
            )
            .def(py::init<>())
            .def(
                "get_action",
                &egttools::FinitePopulations::behaviors::twoActions::Pavlov::get_action,
                R"pbdoc(
                Return the action chosen by the strategy.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    Action selected by the strategy.
            )pbdoc",
                py::arg("time_step"),
                py::arg("action_prev")
            )
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::Pavlov::type,
                 "Return the strategy type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::Pavlov::is_stochastic,
                 "Return whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::ActionInertia,
                egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(
                mNFTwoActions, "ActionInertia",
                "Strategy that tends to repeat its previous action."
            )
            .def(
                py::init<double, double>(),
                R"pbdoc(
                Construct an ActionInertia strategy.

                The strategy repeats its current action, but explores a different action
                with probability `epsilon`. In the first round it cooperates with probability `p`.

                Parameters
                ----------
                epsilon : float
                    Probability of changing action.
                p : float
                    Probability of cooperation in the first round.
            )pbdoc",
                py::arg("epsilon"),
                py::arg("p")
            )
            .def(
                "get_action",
                &egttools::FinitePopulations::behaviors::twoActions::ActionInertia::get_action,
                R"pbdoc(
                Return the action chosen by the strategy.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    Action selected by the strategy.
            )pbdoc",
                py::arg("time_step"),
                py::arg("action_prev")
            )
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::ActionInertia::type,
                 "Return the strategy type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::ActionInertia::is_stochastic,
                 "Return whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer,
                egttools::FinitePopulations::behaviors::AbstractCRDStrategy>(
                mCRD, "CRDMemoryOnePlayer",
                "Memory-one strategy for collective-risk dilemma games."
            )
            .def(
                py::init<int, int, int, int, int>(),
                R"pbdoc(
                Construct a memory-one strategy for a collective-risk dilemma.

                The strategy contributes `initial_action` in the first round. In later rounds,
                it compares the sum of contributions of the other group members in the previous
                round to `personal_threshold`.

                - if the previous group contribution is greater than `personal_threshold`,
                  the player chooses `action_above`;
                - if it is equal to `personal_threshold`, the player chooses `action_equal`;
                - if it is smaller than `personal_threshold`, the player chooses `action_below`.

                Parameters
                ----------
                personal_threshold : int
                    Threshold against which the previous group contribution is compared.
                initial_action : int
                    Contribution in the first round.
                action_above : int
                    Contribution used when the previous group contribution is above the threshold.
                action_equal : int
                    Contribution used when the previous group contribution equals the threshold.
                action_below : int
                    Contribution used when the previous group contribution is below the threshold.

                See Also
                --------
                egttools.behaviors.CRD.AbstractCRDStrategy
                egttools.games.CRDGame
                egttools.games.CRDGameTU
            )pbdoc",
                py::arg("personal_threshold"),
                py::arg("initial_action"),
                py::arg("action_above"),
                py::arg("action_equal"),
                py::arg("action_below")
            )
            .def(
                "get_action",
                &egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer::get_action,
                R"pbdoc(
                Return the action chosen by the strategy.

                Parameters
                ----------
                time_step : int
                    Current round.
                group_contributions_prev : int
                    Sum of contributions of the other group members in the previous round.

                Returns
                -------
                int
                    Action selected by the strategy.

                Examples
                --------
                >>> from egttools.behaviors.CRD import CRDMemoryOnePlayer
                >>> strategy = CRDMemoryOnePlayer(4, 2, 4, 2, 0)
                >>> strategy.get_action(0, 0)
                2
            )pbdoc",
                py::arg("time_step"),
                py::arg("group_contributions_prev")
            )
            .def("type", &egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer::type,
                 "Return the strategy type.")
            .def("__str__", &egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer::toString);
}
