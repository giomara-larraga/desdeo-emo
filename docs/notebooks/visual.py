import plotly.graph_objects as go
from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors


def visualize_2D_front_rvs(front, vectors: ReferenceVectors):
    fig = go.Figure(
        data=go.Scatter(
            x=front[:, 0],
            y=front[:, 1],
            name="Composite front",
            mode="markers",
            marker_size=3,
        )
    )

    for i in range(0, vectors.number_of_vectors):
        fig.add_trace(
            go.Scatter(
                x=[0, vectors.values[i, 0], vectors.values[i, 0]],
                y=[0, vectors.values[i, 1], vectors.values[i, 1]],
                name="vector #" + str(i + 1),
                marker=dict(size=1, opacity=0.8),
                line=dict(width=2),
            )
        )
    return fig


def visualize_3D_front_rvs(front, vectors: ReferenceVectors):

    fig = go.Figure(
        data=go.Scatter3d(
            x=front[:, 0],
            y=front[:, 1],
            z=front[:, 2],
            name="Composite front",
            mode="markers",
            marker_size=3,
        )
    )

    for i in range(0, vectors.number_of_vectors):
        fig.add_trace(
            go.Scatter3d(
                x=[0, vectors.values[i, 0], vectors.values[i, 0]],
                y=[0, vectors.values[i, 1], vectors.values[i, 1]],
                z=[0, vectors.values[i, 2], vectors.values[i, 2]],
                name="vector #" + str(i + 1),
                marker=dict(size=1, opacity=0.8),
                line=dict(width=2),
            )
        )
    return fig


def visualize_2D_front_rp(front, rp):
    fig = go.Figure(
        data=go.Scatter(
            x=front[:, 0],
            y=front[:, 1],
            name="Composite front",
            mode="markers",
            marker_size=3,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[rp[0]], y=[rp[1]], name="Reference point", mode="markers", marker_size=5,
        )
    )
    return fig


def visualize_3D_front_ps(front, ps):
    fig = go.Figure(
        data=go.Scatter3d(
            x=front[:, 0],
            y=front[:, 1],
            z=front[:, 2],
            name="Composite front",
            mode="markers",
            marker_size=3,
        )
    )

    for i in range(0, ps.shape[0]):
        fig.add_trace(
            go.Scatter3d(
                x=[ps[i, 0]],
                y=[ps[i, 1]],
                z=[ps[i, 2]],
                name="preferred solution #" + str(i + 1),
                mode="markers",
                marker_size=5,
            )
        )
    return fig


def visualize_3D_front_rp(front, rp):
    fig = go.Figure(
        data=go.Scatter3d(
            x=front[:, 0],
            y=front[:, 1],
            z=front[:, 2],
            name="Composite front",
            mode="markers",
            marker_size=3,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[rp[0]],
            y=[rp[1]],
            z=[rp[2]],
            name="Reference point",
            mode="markers",
            marker_size=5,
        )
    )
    return fig


def visualize_2D_front_range(front, range, rp):

    fig = go.Figure(
        data=go.Scatter(
            x=front[:, 0],
            y=front[:, 1],
            name="Composite front",
            mode="markers",
            marker_size=3,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[range[0, 0], range[0, 1]],
            y=[rp[1], rp[1]],
            name="preferred range X",
            marker=dict(size=1, opacity=0.8),
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[rp[0], rp[0]],
            y=[range[1, 0], range[1, 1]],
            name="preferred range Y",
            marker=dict(size=1, opacity=0.8),
            line=dict(width=2),
        )
    )

    return fig


def visualize_3D_front_range(front, range, rp):
    fig = go.Figure(
        data=go.Scatter3d(
            x=front[:, 0],
            y=front[:, 1],
            z=front[:, 2],
            name="Composite front",
            mode="markers",
            marker_size=3,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[range[0, 0], range[0, 1]],
            y=[rp[1], rp[1]],
            z=[rp[2], rp[2]],
            name="preferred range X",
            marker=dict(size=1, opacity=0.8),
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[rp[0], rp[0]],
            y=[range[1, 0], range[1, 1]],
            z=[rp[2], rp[2]],
            name="preferred range Y",
            marker=dict(size=1, opacity=0.8),
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[rp[0], rp[0]],
            y=[rp[1], rp[1]],
            z=[range[2, 0], range[2, 1]],
            name="preferred range Z",
            marker=dict(size=1, opacity=0.8),
            line=dict(width=2),
        )
    )
    return fig
