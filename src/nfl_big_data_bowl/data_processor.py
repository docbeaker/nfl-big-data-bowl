import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm


class DataProcessor:
    PLAYIDS = ["gameId", "playId"]
    CARRY_START_EVENTS = {"run", "handoff", "pass_outcome_caught", "snap_direct"}

    @classmethod
    def construct_ball_carrier_view(cls, plays, tracking):
        """
        Construct a view of tracking data focused on ball carriers,
        subset to the frames where they have the ball
        """
        ball_carriers = tracking.merge(
            plays[cls.PLAYIDS + ["ballCarrierId"]],
            left_on=cls.PLAYIDS + ["nflId"],
            right_on=cls.PLAYIDS + ["ballCarrierId"],
        )
        # take last non-NA event as end of play
        play_end = (
            ball_carriers.dropna(subset="event")
            .drop_duplicates(subset=cls.PLAYIDS, keep="last")[
                cls.PLAYIDS + ["frameId", "event", "x"]
            ]
            .rename(columns=dict(frameId="endFrameId", event="endEvent", x="x_final"))
        )
        # consider class of events that are valid for ball carrier to have received the ball
        carry_start = (
            ball_carriers[
                ball_carriers.event.isin(
                    {"run", "handoff", "pass_outcome_caught", "snap_direct"}
                )
            ]
            .drop_duplicates(subset=cls.PLAYIDS, keep="last")[
                cls.PLAYIDS + ["frameId", "event"]
            ]
            .rename(columns=dict(frameId="startFrameId", event="receiptEvent"))
        )
        carry_window = carry_start.merge(play_end, on=cls.PLAYIDS)

        # extract only those frames between ball receipt and end of play
        ball_carriers = ball_carriers.merge(carry_window, on=cls.PLAYIDS)

        angle_to_downfield = (
            np.pi
            * np.where(
                ball_carriers.playDirection == "right",
                ball_carriers.dir - 90,
                ball_carriers.dir - 270,
            )
            / 180
        )
        ball_carriers["s_downfield"] = ball_carriers.s * np.cos(angle_to_downfield)

        return (
            ball_carriers[
                ball_carriers.frameId.between(
                    ball_carriers.startFrameId, ball_carriers.endFrameId
                )
            ]
            .reset_index(drop=True)
            .astype({"nflId": "int", "jerseyNumber": "int"})
        )

    @classmethod
    def construct_defender_view(cls, carriers, tracking, nearby_radius=5):
        """
        Given a carrier view, determine where the defenders are relative to the carrier
        """
        mergeby = cls.PLAYIDS + ["frameId"]

        # Right now really "all others" instead of defenders, but we're going to drop
        # the same-team players shortly, so just use _defenders as a suffix for now
        others = carriers.merge(
            tracking.drop(columns=["playDirection", "time", "event"]),
            on=mergeby,
            suffixes=("_carrier", "_defender"),
        )
        others.nflId_defender = others.nflId_defender.fillna(0).astype("int")
        others["dist_to_carrier"] = np.sqrt(
            np.power(others.x_carrier - others.x_defender, 2)
            + np.power(others.y_carrier - others.y_defender, 2)
        )

        # Compute how many same team players are nearby
        support = others[
            (others.club_defender == others.club_carrier)
            & (others.nflId_defender != others.nflId_carrier)
        ]
        support_nearby = (
            support[support.dist_to_carrier < nearby_radius]
            .groupby(mergeby)
            .nflId_defender.nunique()
            .rename("blockers_nearby")
        )
        n_pre_merge = len(others)
        others = others.merge(
            support_nearby, left_on=mergeby, right_index=True, how="left"
        )
        others.blockers_nearby = others.blockers_nearby.fillna(0).astype("int")
        assert (
            len(others) == n_pre_merge
        ), f"lost rows in carrier support merge: {len(others)} vs {n_pre_merge}"

        defenders = (
            others[
                (others.club_defender != others.club_carrier)
                & (others.club_defender != "football")
            ]
            .reset_index(drop=True)
            .astype({"jerseyNumber_defender": "int"})
        )
        defenders_nearby = (
            defenders[defenders.dist_to_carrier < nearby_radius]
            .groupby(mergeby)
            .nflId_defender.nunique()
            .rename("defenders_nearby")
        )
        n_pre_merge = len(defenders)
        defenders = defenders.merge(
            defenders_nearby, left_on=mergeby, right_index=True, how="left"
        )
        defenders.defenders_nearby = defenders.defenders_nearby.fillna(0).astype("int")
        assert (
            len(defenders) == n_pre_merge
        ), f"lost rows in defender support merge: {len(defenders)} vs {n_pre_merge}"

        return defenders

    @classmethod
    def add_physical_characteristics(cls, defenders, players):
        """
        Add some data about the defender and carrier involved
        """
        height = players.height.str.split("-")
        players["height_m"] = (
            height.str[0].astype("int") + height.str[1].astype("int") / 12
        ) * 0.3048
        players = players.set_index("nflId")[["weight", "position", "height_m"]]
        defenders = defenders.merge(players, left_on="nflId_carrier", right_index=True)
        defenders = defenders.merge(
            players,
            left_on="nflId_defender",
            right_index=True,
            suffixes=("_carrier", "_defender"),
        )

        return defenders

    @classmethod
    def add_labels(cls, defenders, tackles):
        defenders = defenders.merge(
            tackles,
            left_on=DataProcessor.PLAYIDS + ["nflId_defender"],
            right_on=DataProcessor.PLAYIDS + ["nflId"],
            how="left",
        ).drop(columns="nflId")
        map_cols = [c for c in tackles if c != "nflId"]
        defenders[map_cols] = defenders[map_cols].fillna(0).astype("int")
        return defenders

    @classmethod
    def pipeline(cls, plays, players, tackles, tracking_dfs):
        defenders = []
        for _df in tqdm(tracking_dfs):
            if isinstance(_df, Path):
                _df = pd.read_csv(_df)
            carriers = DataProcessor.construct_ball_carrier_view(plays, _df)
            _defenders = DataProcessor.construct_defender_view(carriers, _df)
            _defenders = DataProcessor.add_physical_characteristics(_defenders, players)
            _defenders = DataProcessor.add_labels(_defenders, tackles)
            defenders.append(_defenders)
        defenders = pd.concat(defenders).reset_index(drop=True)
        return defenders.sort_values(
            cls.PLAYIDS + ["frameId", "nflId_defender"]
        ).reset_index(drop=True)
