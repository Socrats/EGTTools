/** Copyright (c) 2022-2023  Elias Fernandez
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
    std::string call_get_action(const py::list &strategies, size_t time_step, int action) {
        std::stringstream result;
        result << "(";
        for (py::handle strategy : strategies) {
            result << py::cast<egttools::FinitePopulations::behaviors::AbstractNFGStrategy *>(strategy)->get_action(time_step, action) << ", ";
        }
        result << ")";
        return result.str();
    }
}// namespace egttools

void init_behaviors(py::module_ &mBehaviors) {
    auto mCRD = mBehaviors.def_submodule("CRD");
    auto mNF = mBehaviors.def_submodule("NormalForm");
    auto mNFTwoActions = mNF.def_submodule("TwoActions");

    mBehaviors.attr("__init__") = py::str("The `egttools.numerical.behaviors` submodule contains the available strategies to evolve.");
    mNF.attr("__init__") = py::str("The `egttools.numerical.behaviors.NormalForm` submodule contains the strategies for normal form games.");
    mCRD.attr("__init__") = py::str("The `egttools.numerical.behaviors.CRD` submodule contains the strategies for the CRD.");
    mNFTwoActions.attr("__init__") = py::str(
            "The `TwoActions` submodule contains the strategies for "
            "normal form games with 2 actions.");

    mBehaviors.def("call_get_action", &egttools::call_get_action,
                   "Returns a string with a tuple of actions",
                   py::arg("strategies"),
                   py::arg("time_step"),
                   py::arg("prev_action"));

    py::class_<egttools::FinitePopulations::behaviors::AbstractNFGStrategy, stubs::PyAbstractNFGStrategy>(mNF, "AbstractNFGStrategy")
            .def(py::init<>())
            .def("get_action", &egttools::FinitePopulations::behaviors::AbstractNFGStrategy::get_action,
                 R"pbdoc(
                Returns an action in function of time_step round and the previous action action_prev of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.behaviors.NormalForm.TwoActions.Cooperator, egttools.behaviors.NormalForm.TwoActions.Defector,
                egttools.behaviors.NormalForm.TwoActions.Random, egttools.behaviors.NormalForm.TwoActions.TFT,
                egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT, egttools.behaviors.NormalForm.TwoActions.GenerousTFT,
                egttools.behaviors.NormalForm.TwoActions.GradualTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::AbstractNFGStrategy::type, "Returns a string indicating the Strategy Type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::AbstractNFGStrategy::is_stochastic,
                 "Property indicating if the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::AbstractCRDStrategy, stubs::PyAbstractCRDStrategy>(mCRD, "AbstractCRDStrategy")
            .def(py::init<>())
            .def("get_action", &egttools::FinitePopulations::behaviors::AbstractCRDStrategy::get_action,
                 R"pbdoc(
                Returns an action in function of time_step round and the previous action action_prev of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                group_contributions_prev : int
                    Sum of contributions of the other members of the group (excluding the focal player) in the previous round.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.behaviors.CRD.CRDMemoryOnePlayer,
                egttools.behaviors.NormalForm.TwoActions.Cooperator, egttools.behaviors.NormalForm.TwoActions.Defector,
                egttools.behaviors.NormalForm.TwoActions.Random, egttools.behaviors.NormalForm.TwoActions.TFT,
                egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT, egttools.behaviors.NormalForm.TwoActions.GenerousTFT,
                egttools.behaviors.NormalForm.TwoActions.GradualTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("group_contributions_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::AbstractCRDStrategy::type, "Returns a string indicating the Strategy Type.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::Cooperator,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "Cooperator")
            .def(py::init<>(), "This strategy always cooperates.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::Cooperator::get_action,
                 R"pbdoc(
                Returns an action in function of time_step round and the previous action action_prev of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Defector,
                egttools.behaviors.NormalForm.TwoActions.Random, egttools.behaviors.NormalForm.TwoActions.TFT,
                egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT, egttools.behaviors.NormalForm.TwoActions.GenerousTFT,
                egttools.behaviors.NormalForm.TwoActions.GradualTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::Cooperator::type, "Returns a string indicating the Strategy Type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::Cooperator::is_stochastic,
                 "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::Defector,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "Defector")
            .def(py::init<>(), "This strategy always defects.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::Defector::get_action,
                 R"pbdoc(
                Returns an action in function of time_step round and the previous action action_prev of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Random, egttools.behaviors.NormalForm.TwoActions.TFT,
                egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT, egttools.behaviors.NormalForm.TwoActions.GenerousTFT,
                egttools.behaviors.NormalForm.TwoActions.GradualTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::Defector::type, "Returns a string indicating the Strategy Type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::Defector::is_stochastic,
                 "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::RandomPlayer,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "Random")
            .def(py::init<>(), "This players chooses cooperation with uniform random probability.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::get_action,
                 R"pbdoc(
                Returns an action in function of time_step round and the previous action action_prev of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.TFT,
                egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT, egttools.behaviors.NormalForm.TwoActions.GenerousTFT,
                egttools.behaviors.NormalForm.TwoActions.GradualTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::type, "Returns a string indicating the Strategy Type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::is_stochastic,
                 "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::TitForTat,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "TFT")
            .def(py::init<>(), "Tit for Tat: Cooperates in the first round and imitates the opponent's move thereafter.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::TitForTat::get_action,
                 R"pbdoc(
                Returns an action in function of time_step round and the previous action action_prev of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT, egttools.behaviors.NormalForm.TwoActions.GenerousTFT,
                egttools.behaviors.NormalForm.TwoActions.GradualTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::TitForTat::type, "Returns a string indicating the Strategy Type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::TitForTat::is_stochastic,
                 "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "SuspiciousTFT")
            .def(py::init<>(), "Defects on the first round and imitates its opponent's previous move thereafter.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::get_action,
                 R"pbdoc(
                Returns an action in function of time_step round and the previous action action_prev of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.GenerousTFT,
                egttools.behaviors.NormalForm.TwoActions.GradualTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::type, "Returns a string indicating the Strategy Type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::is_stochastic,
                 "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::GenerousTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "GenerousTFT")
            .def(py::init<double, double, double, double>(),
                 "Cooperates on the first round and after its opponent "
                 "cooperates. Following a defection,it cooperates with probability\n"
                 "@f[ p(R,P,T,S) = min{1 - \\frac{T-R}{R-S}, \\frac{R-P}{T-P}} @f]\n"
                 "where R, P, T and S are the reward, punishment, temptation and "
                 "suckers payoffs.",
                 py::arg("R"), py::arg("P"),
                 py::arg("T"), py::arg("S"))
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::get_action,
                 R"pbdoc(
                Returns an action in function of time_step round and the previous action action_prev of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GradualTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::type, "Returns a string indicating the Strategy Type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::is_stochastic,
                 "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::GradualTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "GradualTFT")
            .def(py::init<>(),
                 "TFT with two differences:\n"
                 "(1) it increases the string of punishing defection responses "
                 "with each additional defection by its opponent\n"
                 "(2) it apologizes for each string of defections"
                 "by cooperating in the subsequent two rounds.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::GradualTFT::get_action,
                 R"pbdoc(
                Returns an action in function of time_step round and the previous action action_prev of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.ImperfectTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::GradualTFT::type, "Returns a string indicating the Strategy Type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::GradualTFT::is_stochastic,
                 "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "ImperfectTFT")
            .def(py::init<double>(),
                 "Imitates opponent as in TFT, but makes mistakes "
                 "with :param error_probability.",
                 py::arg("error_probability"))
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::get_action,
                 R"pbdoc(
                Returns an action in function of time_step round and the previous action action_prev of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.GradualTFT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::type, "Returns a string indicating the Strategy Type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::is_stochastic,
                 "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::TFTT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "TFTT")
            .def(py::init<>(), "Tit for 2 tats: Defects if defected twice.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::TFTT::get_action,
                 R"pbdoc(
                Returns an action in function of time_step round and the previous action action_prev of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.GradualTFT,
                egttools.behaviors.NormalForm.TwoActions.ImperfectTFT, egttools.behaviors.NormalForm.TwoActions.TTFT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::TFTT::type, "Returns a string indicating the Strategy Type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::TFTT::is_stochastic,
                 "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::TTFT,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "TTFT")
            .def(py::init<>(), "2 Tits for tat: Defects twice if defected.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::TTFT::get_action,
                 R"pbdoc(
                Returns an action in function of time_step round and the previous action action_prev of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.GradualTFT,
                egttools.behaviors.NormalForm.TwoActions.ImperfectTFT, egttools.behaviors.NormalForm.TwoActions.TFTT,
                egttools.behaviors.NormalForm.TwoActions.GRIM, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::TTFT::type, "Returns a string indicating the Strategy Type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::TTFT::is_stochastic,
                 "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::GRIM,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "GRIM")
            .def(py::init<>(),
                 "Grim (Trigger): Cooperates until its opponent has defected once, "
                 "and then defects for the rest of the game.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::GRIM::get_action,
                 R"pbdoc(
                Returns an action in function of time_step round and the previous action action_prev of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.GradualTFT,
                egttools.behaviors.NormalForm.TwoActions.ImperfectTFT, egttools.behaviors.NormalForm.TwoActions.TFTT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.Pavlov
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::GRIM::type, "Returns a string indicating the Strategy Type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::GRIM::is_stochastic,
                 "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::Pavlov,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "Pavlov")
            .def(py::init<>(),
                 "Win-stay loose-shift: Cooperates if it and its opponent moved alike in"
                 "previous move and defects if they moved differently.")
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::Pavlov::get_action,
                 R"pbdoc(
                Returns an action in function of time_step round and the previous action action_prev of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.GradualTFT,
                egttools.behaviors.NormalForm.TwoActions.ImperfectTFT, egttools.behaviors.NormalForm.TwoActions.TFTT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.GRIM
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::Pavlov::type, "Returns a string indicating the Strategy Type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::Pavlov::is_stochastic,
                 "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::twoActions::ActionInertia,
               egttools::FinitePopulations::behaviors::AbstractNFGStrategy>(mNFTwoActions, "ActionInertia")
            .def(py::init<double, double>(),
                 R"pbdoc(
                Always repeats the same action, but explores a different action with probability :param epsilon.

                In the first round it will cooperate with probability :param p.

                Parameters
                ----------
                epsilon : double
                    Probability of changing action.
                p : double
                    Probability of cooperation in the first round

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.GradualTFT,
                egttools.behaviors.NormalForm.TwoActions.ImperfectTFT, egttools.behaviors.NormalForm.TwoActions.TFTT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.GRIM
                )pbdoc",
                 py::arg("epsilon"), py::arg("p"))
            .def("get_action", &egttools::FinitePopulations::behaviors::twoActions::ActionInertia::get_action,
                 R"pbdoc(
                Returns an action in function of time_step round and the previous action action_prev of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                action_prev : int
                    Previous action of the opponent.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.games.AbstractGame, egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.GradualTFT,
                egttools.behaviors.NormalForm.TwoActions.ImperfectTFT, egttools.behaviors.NormalForm.TwoActions.TFTT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.GRIM
                )pbdoc",
                 py::arg("time_step"), py::arg("action_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::twoActions::ActionInertia::type, "Returns a string indicating the Strategy Type.")
            .def("is_stochastic", &egttools::FinitePopulations::behaviors::twoActions::ActionInertia::is_stochastic,
                 "Indicates whether the strategy is stochastic.");

    py::class_<egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer,
               egttools::FinitePopulations::behaviors::AbstractCRDStrategy>(mCRD, "CRDMemoryOnePlayer")
            .def(py::init<int, int, int, int, int>(),
                 R"pbdoc(
                This strategy contributes in function of the contributions of the rest of the group in the previous round.

                This strategy contributes @param initial_action in the first round of the game,
                afterwards compares the sum of contributions of the other members of the group
                in the previous round (a_{-i}(t-1)) to a :param personal_threshold. If the
                a_{-i}(t-1)) > personal_threshold the agent contributions :param action_above,
                if a_{-i}(t-1)) = personal_threshold it contributes :param action_equal
                or if a_{-i}(t-1)) < personal_threshold it contributes :param action_below.

                Parameters
                ----------
                personal_threshold : int
                    threshold value compared to the contributions of the other members of the group
                initial_action : int
                    Contribution in the first round
                action_above : int
                    contribution if a_{-i}(t-1)) > personal_threshold
                action_equal : int
                    contribution if a_{-i}(t-1)) = personal_threshold
                action_below : int
                    contribution if a_{-i}(t-1)) < personal_threshold

                See Also
                --------
                egttools.behaviors.CRD.AbstractCRDStrategy
                egttools.games.AbstractGame,
                egttools.games.CRDGame,
                egttools.games.CRDGameTU
                )pbdoc",
                 py::arg("personal_threshold"), py::arg("initial_action"),
                 py::arg("action_above"), py::arg("action_equal"), py::arg("action_below"))
            .def("get_action", &egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer::get_action,
                 R"pbdoc(
                Returns an action in function of time_step round and the previous action action_prev of the opponent.

                Parameters
                ----------
                time_step : int
                    Current round.
                group_contributions_prev : int
                    Sum of contributions of the other members of the group (without
                    the focal player) in the previous round.

                Returns
                -------
                int
                    The action selected by the strategy.

                See Also
                --------
                egttools.behaviors.CRD.AbstractCRDStrategy
                egttools.games.AbstractGame,
                egttools.games.CRDGame,
                egttools.games.CRDGameTU,
                egttools.behaviors.NormalForm.TwoActions.Cooperator,
                egttools.behaviors.NormalForm.TwoActions.Defector, egttools.behaviors.NormalForm.TwoActions.Random,
                egttools.behaviors.NormalForm.TwoActions.TFT, egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT,
                egttools.behaviors.NormalForm.TwoActions.GenerousTFT, egttools.behaviors.NormalForm.TwoActions.GradualTFT,
                egttools.behaviors.NormalForm.TwoActions.ImperfectTFT, egttools.behaviors.NormalForm.TwoActions.TFTT,
                egttools.behaviors.NormalForm.TwoActions.TFTT, egttools.behaviors.NormalForm.TwoActions.GRIM
                )pbdoc",
                 py::arg("time_step"), py::arg("group_contributions_prev"))
            .def("type", &egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer::type, "Returns a string indicating the Strategy Type.")
            .def("__str__", &egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer::toString);
}