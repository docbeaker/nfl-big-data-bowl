import numpy as np

class TackleFeatures:
    WEIGHT_NORM = 250  # Scale by approximate mean player weight to keep values sensible
    FWIDTH = 53.3
    PLAYDEF = ["gameId", "playId", "nflId_defender"]

    @classmethod
    def relative_angle(cls, theta_carrier, theta_defender):
        theta_rel = theta_defender - theta_carrier
        return (
            np.where(
                theta_rel.between(-180, 180), np.abs(theta_rel), 360 - np.abs(theta_rel)
            )
            / 180
        )

    @classmethod
    def relative_vector_magnitude(cls, s_carrier, s_defender, theta_rel):
        return np.sqrt(
            np.power(s_carrier, 2)
            + np.power(s_defender, 2)
            - 2 * s_carrier * s_defender * np.cos(np.pi * theta_rel)
        )

    @classmethod
    def add_tackle_features(cls, defenders):
        existing_features = ["s_carrier"]

        defenders["p_carrier"] = defenders.s_carrier * (
            defenders.weight_carrier / cls.WEIGHT_NORM
        )
        p_defender = defenders.s_defender * (
            defenders.weight_defender / cls.WEIGHT_NORM
        )
        dir_rel = cls.relative_angle(defenders.dir_carrier, defenders.dir_defender)

        defenders["p_rel"] = cls.relative_vector_magnitude(
            defenders.p_carrier, p_defender, dir_rel
        )
        defenders["o_rel"] = cls.relative_angle(
            defenders.o_carrier, defenders.o_defender
        )

        defenders["rel_height"] = (
            defenders.height_m_defender / defenders.height_m_carrier
        )

        # Frames appear to be about 1ms apart, so just use that to avoid datetime nonsense
        defenders["time_since_receipt"] = 0.1 * (
            defenders.frameId - defenders.startFrameId
        )

        defenders["sideline_dist"] = (
            np.minimum(defenders.y_carrier, cls.FWIDTH - defenders.y_carrier)
            / cls.FWIDTH
            * 2
        )

        defenders["is_caught_pass"] = (
            defenders.receiptEvent == "pass_outcome_caught"
        ).astype("int")

        defenders["is_qb"] = (defenders.position_carrier == "QB").astype("int")

        defenders["net_defenders_nearby"] = (
            defenders.defenders_nearby - defenders.blockers_nearby
        )
        defenders["congestion"] = defenders.defenders_nearby + defenders.blockers_nearby

        new_f_idx = defenders.columns.get_loc("p_carrier")
        return existing_features + defenders.columns.tolist()[new_f_idx:]

    @classmethod
    def extract_tackle_opps(cls, defenders, tackle_dist=1.5):
        # Extract all defenders get closer enough to the defender to potentially
        # make a tackle
        dsorted = (
            defenders[
                (defenders.dist_to_carrier <= tackle_dist)
                & defenders[["tackle", "assist", "pff_missedTackle"]].any(axis=1)
            ]
            .sort_values(cls.PLAYDEF + ["frameId"])
            .reset_index(drop=True)
        )

        # A bit of trickery to assign a "tackle opportunity index" = tidx to each carrier/defender interaction.
        # Note that, in some cases, there can be multiple such events in the same play if
        # the defender gets father and then closer. We will detect this via
        # "skips" in frames (as we've already focused only on frames where the two are close)
        dsorted["tidx"] = (
            dsorted[cls.PLAYDEF].diff().any(axis=1) | (dsorted.frameId.diff() != 1)
        ).cumsum()

        # We only care about the state at the beginning of each opportunity
        dsorted = dsorted.drop_duplicates("tidx", keep="first")

        # Logging: how many events are non-ambiguous
        nopps = dsorted.groupby(cls.PLAYDEF).agg(dict(
            tidx="nunique", tackle="max", assist="max", pff_missedTackle="max"
        ))
        non_ambig = (
            (nopps.tidx == 1) & (nopps[["tackle", "assist", "pff_missedTackle"]].sum(axis=1) == 1)
        ).mean()
        print(f"proportion of tackle opportunities that are non-ambiguous = {100*non_ambig:.1f}%")

        # Assume tackles & assists only happen on the last opportunity
        final_opp_idx = dsorted.groupby(cls.PLAYDEF).tidx.max().rename("tidx_max")
        dsorted = dsorted.merge(final_opp_idx, left_on=cls.PLAYDEF, right_index=True)
        dsorted[["tackle", "assist"]] = dsorted[["tackle", "assist"]] * (
            dsorted.tidx_max == dsorted.tidx
        ).astype("int").values.reshape((-1, 1))
        # Drop resulting cases where we have an opp without a tackle as it has been reassigned
        dsorted = dsorted[
            dsorted[["tackle", "assist", "pff_missedTackle"]].any(axis=1)
        ].reset_index(drop=True)
        dsorted = dsorted.drop(columns=["tidx_max"])
        # If there's a tackle or assist in the opportunity, we'll say there's not a miss
        dsorted.pff_missedTackle = dsorted.pff_missedTackle * (
            1 - dsorted[["tackle", "assist"]].sum(axis=1)
        )
        # Note: could also condition missed tackles must come before tackles?

        # Compute weights
        weights = 1 / dsorted.groupby(cls.PLAYDEF).pff_missedTackle.sum().rename(
            "weight"
        )
        weights = weights.fillna(1)
        dsorted = dsorted.merge(weights, left_on=cls.PLAYDEF, right_index=True)
        dsorted.weight = np.where(dsorted.pff_missedTackle == 1, dsorted.weight, 1)

        tackle_features = cls.add_tackle_features(dsorted)
        return dsorted, tackle_features
