import plotly.graph_objects as go
import os
import logging
import plotly.io as pio

class DataVisualizer:
    """
    A class used to visualize data.

    ...

    Static Methods
    -------
    create_plot(data, layout) -> go.Figure
        Creates a plotly figure with the given data and layout.
    visualize_dataframe(df, top_contributors, title, chart_type='bar') -> None
        Visualizes the DataFrame as a bar or pie chart.
    plot_time_series(df, x_column, y_column, label, title) -> None
        Plots a time series from the DataFrame.
    visualize_bar(df, title, xlabel, ylabel) -> None
        Visualizes the DataFrame as a bar chart.
    visualize_pie(values, labels, title) -> None
        Visualizes the data as a pie chart.
    """

    @staticmethod
    def create_plot(data, layout):
        """Creates a plotly figure with the given data and layout."""
        fig = go.Figure(data=data, layout=layout)
        fig.show()
        pio.write_html(fig, '{layout}.html')
        
    @staticmethod
    def visualize_dataframe(df, top_contributors, title, xlabel, ylabel, chart_type='bar'): 
        """Visualizes the DataFrame as a bar or pie chart."""
        try:
            mean = df['FinalValue'].mean()
            median = df['FinalValue'].median()
            std_dev = df['FinalValue'].std()
            # Create custom hover text
            hover_text = ['{}: {}<br>{}: {:.2f}<br>Mean: {:.2f}<br>Median: {:.2f}<br>Std Dev: {:.2f}'.format(xlabel, x, ylabel, y, mean, median, std_dev) for x, y in zip(top_contributors.index, top_contributors)]
            
            layout = go.Layout(title_text=title)
            if chart_type == 'bar':
                data = [go.Bar(x=top_contributors.index, y=top_contributors, text=hover_text, textposition='auto', hoverinfo='text')]
            elif chart_type == 'pie':
                data = [go.Pie(labels=top_contributors.index, values=top_contributors)]
            # Add more chart types here as needed
            DataVisualizer.create_plot(data, layout)
        except Exception as e:
            logging.error("Error in visualize_dataframe method: %s", str(e))
            raise

    @staticmethod
    def plot_time_series(df, x_column, y_column, label, title):
        """Plots a time series from the DataFrame."""
        try:
            data = [go.Scatter(x=df[x_column], y=df[y_column], mode='lines', name=label)]
            layout = go.Layout(title_text=title)
            DataVisualizer.create_plot(data, layout)
        except Exception as e:
            logging.error("Error in plot_time_series method: %s", str(e))
            raise

    @staticmethod
    def visualize_bar(df, title, xlabel, ylabel):
        """Visualizes the DataFrame as a bar chart."""
        try:
            mean = df.mean()
            median = df.median()
            std_dev = df.std()
            # Create custom hover text
            hover_text = ['{}: {}<br>{}: {:.2f}<br>Mean: {:.2f}<br>Median: {:.2f}<br>Std Dev: {:.2f}'.format(xlabel, x, ylabel, y, mean, median, std_dev) for x, y in zip(df.index, df)]
            
            data = [go.Bar(x=df.index, y=df, text=hover_text, textposition='auto', hoverinfo='text')]
            layout = go.Layout(title_text=title, xaxis_title=xlabel, yaxis_title=ylabel)
            DataVisualizer.create_plot(data, layout)
        except Exception as e:
            logging.error("Error in visualize_bar method: %s", str(e))
            raise

    @staticmethod
    def visualize_pie(values, labels, title):
        """Visualizes the data as a pie chart."""
        try:
            data = [go.Pie(labels=labels, values=values)]
            layout = go.Layout(title_text=title)
            DataVisualizer.create_plot(data, layout)
        except Exception as e:
            logging.error("Error in visualize_pie method: %s", str(e))
            raise

    @staticmethod
    def visualize_nested_pie(df, outer_col, inner_col, title):
        """Visualizes the data as a nested pie chart."""
        try:
            outer_counts = df[outer_col].value_counts()
            inner_counts = df[inner_col].value_counts()
            labels = outer_counts.index.tolist() + inner_counts.index.tolist()
            parents = [''] * len(outer_counts) + outer_counts.index.tolist() * len(inner_counts)
            values = outer_counts.tolist() + inner_counts.tolist()

            fig = go.Figure(go.Sunburst(labels=labels, parents=parents, values=values))
            fig.update_layout(title_text=title)
            fig.show()
        except Exception as e:
            logging.error("Error in visualize_nested_pie method: %s", str(e))
            raise


