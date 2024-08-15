import streamlit as st 
import re
import pandas as pd
from io import BytesIO
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

st.set_page_config(layout="wide", page_title="PDF to CSV Processor")

st.write("## Bluebeam PDF to CSV and Visual Insights Processor 🔧")
st.write(" #### ETL Pipeline and Behavior Mapping Solutions for Architectural Mapping")
st.sidebar.write("## Upload one or more Bluebeam PDF files 📂")

def process_pdfs(uploaded_file) -> pd.DataFrame:
    # Initialize a list to hold the extracted data
    extracted_data = []

    # Precompile regex patterns
    subj_pattern = re.compile(r'/Subj\((.*?)\)')
    rect_pattern = re.compile(r'/Rect\[(.*?)\]')
    contents_pattern = re.compile(r'<p>([\s\S]*?)<\/p>')
    color_pattern = re.compile(r'/C\[(.*?)\]')
    rt_pattern = re.compile(r'/RT/(.*?)/')

    filename = uploaded_file.name
    time = os.path.splitext(filename)[0].split('-')[2]
    # Format the time as hh:mm
    time = f"{time[:2]}:{time[2:]}" if len(time) == 4 else '00:00'

    # 使用 BytesIO 读取上传文件的内容
    with BytesIO(uploaded_file.read()) as f:
        for line in f:
            try:
                line_str = line.decode('utf-8', errors='ignore')
                if 'PolyLine' in line_str or 'Ellipse' in line_str:
                    # Search for patterns
                    subj_match = subj_pattern.search(line_str)
                    rect_match = rect_pattern.search(line_str)
                    contents_match = contents_pattern.search(line_str)
                    color_match = color_pattern.search(line_str)
                    rt_match = rt_pattern.search(line_str)

                    # Extract and process data
                    subj = subj_match.group(1) if subj_match else ''
                    rect_coords = list(map(float, rect_match.group(1).split())) if rect_match else []
                    cleaned_contents = re.sub(r'<[^>]+>', '', contents_match.group(1)) if contents_match else ''
                    color = list(map(float, color_match.group(1).split())) if color_match else []
                    rt = rt_match.group(1) if rt_match else ''
                    
                    # Compute center coordinates if rect_coords are available
                    if len(rect_coords) == 4:
                        x_coor = round((rect_coords[0] + rect_coords[2]) / 2, 4)
                        y_coor = round((rect_coords[1] + rect_coords[3]) / 2, 4)
                    else:
                        x_coor = y_coor = None

                    # Append data to the list
                    extracted_data.append([
                        subj,
                        rect_coords,
                        cleaned_contents,
                        color,
                        rt,
                        x_coor,
                        y_coor,
                        time
                    ])

            except Exception as e:
                # Handle any exceptions (e.g., decoding issues, regex errors)
                print(f"Error processing line: {e}")

    # Create a DataFrame from the extracted data
    columns = ['Subj', 'Rect', 'Contents', 'Color', 'RT', 'x_coor', 'y_coor', 'time']
    df = pd.DataFrame(extracted_data, columns=columns)

    # Remove duplicates based on 'Subj' and 'RT'
    df = df[~((df['Subj'].str.contains('PolyLine')) & (df['RT'] == 'Group'))]

    return df

def save_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue()

def generate_bm(df: pd.DataFrame, background, selected_times, selected_contents, color_map):
    if background is not None:
        

        background_image = Image.open(background)
        
        # Ensure the image is in RGBA format
        background_image = background_image.convert("RGBA")
        width, height = background_image.size

        # Resize the data points to fit the background size
        width_scale_factor = background_image.width / 792  # 792 is the assumed original width
        height_scale_factor = background_image.height / 612  # 612 is the assumed original height

        df['x_coor'] = df['x_coor'] * width_scale_factor
        df['y_coor'] = df['y_coor'] * height_scale_factor

        bin_size = 8
        x_bin_edges = np.arange(0, width + 1, bin_size)
        y_bin_edges = np.arange(0, height + 1, bin_size)

        max_count = df.groupby(['Contents', pd.cut(df['x_coor'], bins=x_bin_edges, labels=False, right=False), 
                                pd.cut(df['y_coor'], bins=y_bin_edges, labels=False, right=False)]).size().max()
        fig, ax = plt.subplots(figsize=(17, 11))

        # Set the background image
        ax.imshow(background_image, extent=[0, width, 0, height], aspect='auto')
        ax.axis('off')

        # Prepare color palette and mapping


        # Filter data based on selected times and contents

        df_filtered = df[df['time'].isin(selected_times) & df['Contents'].isin(selected_contents)]

        added_labels = set()

        # Plot each data type
        for type_name in selected_contents:
            non_zero_bins, circle_sizes, type_name = process_data_for_type(df_filtered, type_name, x_bin_edges, y_bin_edges, max_count)
            color = color_map.get(type_name, 'gray')
            for index, row in non_zero_bins.iterrows():
                x_center = (row['x_bin'] * bin_size) + bin_size / 2
                y_center = (row['y_bin'] * bin_size) + bin_size / 2
                if type_name not in added_labels:
                    ax.scatter(x_center, y_center, s=circle_sizes.iloc[index], alpha=1, color=color, label=type_name)
                    added_labels.add(type_name)
                else:
                    ax.scatter(x_center, y_center, s=circle_sizes.iloc[index], alpha=1, color=color)

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        plt.legend(loc='upper center', bbox_to_anchor=(.5, 0.1), ncol=7)

        # Display the plot in Streamlit
        st.pyplot(fig)

def process_data_for_type(df, type_name, x_bin_edges, y_bin_edges, max_count):
    type_df = df[df['Contents'] == type_name].copy()
    type_df['x_bin'] = pd.cut(type_df['x_coor'], bins=x_bin_edges, labels=False, right=False)
    type_df['y_bin'] = pd.cut(type_df['y_coor'], bins=y_bin_edges, labels=False, right=False)
    bin_counts = type_df.groupby(['x_bin', 'y_bin']).size().reset_index(name='count')
    circle_sizes = (bin_counts['count'] / max_count) * 80  # Adjust size relative to max count
    return bin_counts[bin_counts['count'] > 0], circle_sizes, type_name

def pie_chart(df, selected_contents, color_map):
    # 每个content的percentage在selected content里，全部时间段
    # The percentage of each content type within the selected contents across all time periods.
    df_selected_contents = df[df['Contents'].isin(selected_contents)]
    content_count = df_selected_contents['Contents'].value_counts()

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = [color_map.get(content, 'gray') for content in content_count.index]
    ax.pie(content_count, labels=content_count.index, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig)

def line_chart(df, selected_times, color_map):
    df_selected_contents = df[df['time'].isin(selected_times)]
    selected_count = df_selected_contents['time'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(selected_count.index, selected_count.values, marker='o', color=color_map.get('line_color', 'b'))
    ax.set_title('Count of Occurrences Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


def main():
    tab1, tab2 = st.tabs(["Introduction", "Behavior Mapping"])
    with tab1:
        st.markdown("""
        ### Welcome to the Bluebeam PDF to CSV and Visual Insights Processor!

        This application is designed to streamline the process of converting complex PDF data into insightful visualizations. Whether you’re working with data extracted from architectural plans, engineering diagrams, or other detailed documents, this app simplifies the task of analyzing and visualizing that data.

        ### How to Use
        1. **Upload your PDF file.** The app will process the PDF and extract data into a DataFrame.
        2. **Select times and contents.** Use the sidebar to filter the data according to specific times and content types.
        3. **Generate visualizations.** The app will create visualizations based on your selections, including pie charts and line charts.
        4. **Download data.** Use the download button to save the processed data as a CSV file.

        ### Behavior Mapping
        - **Dots on the Map**: Each dot represents a data point from your PDF, plotted according to its coordinates. The size of each dot indicates the frequency of occurrences within a specific area.
        - **Color Coding**: Different colors represent different types of activities or contents, as defined by the `Contents` column in your data. Each type is assigned a unique color for easy identification.
        - **Time Filtering**: By selecting specific times, you can focus on data points collected during those time periods. This allows you to analyze changes or trends over different times.
        - **Content Filtering**: You can also filter by content types to view only the data points relevant to selected categories. This helps in analyzing specific types of activities or items.

        The generated visualizations provide a visual summary of your data, helping you to quickly identify patterns, trends, and areas of interest based on the filters you apply. Larger dots indicate more frequent occurrences of a particular activity or content type, while the color helps you distinguish between different types of activities.

        Feel free to explore different combinations of times and contents to gain deeper insights into your data!
        """)

    with tab2:
        st.markdown("### Behavior Mapping")
        result = st.sidebar.file_uploader('Upload PDF files', type='pdf', accept_multiple_files=True)
        st.sidebar.write("## Upload your clean floorplan :gear:")
        background = st.sidebar.file_uploader('Upload Floorplan', type=["png", "jpg", "jpeg"])

        if result:
            all_data = []
            for file in result:
                file_data = process_pdfs(file)
                all_data.append(file_data)

            combined_df = pd.concat(all_data, ignore_index=True)

            types = combined_df['Contents'].unique()
            time_options = combined_df['time'].unique()
            selected_times = st.sidebar.multiselect('Select Times:', options=time_options)
            selected_contents = st.sidebar.multiselect('Select Contents:', options=types)
            colors = sns.color_palette("tab20", len(types))
            color_map = {subject: color for subject, color in zip(types, colors)}
            
            st.write('Visualization display here')
            csv_data = save_csv(combined_df)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="processed_data.csv",
                mime='text/csv'
            )
            

            if background:
                bm_fig = generate_bm(combined_df, background, selected_times, selected_contents, color_map)
                st.pyplot(bm_fig)
            else:
                st.warning("Please upload a floorplan to generate the visualization.")
            
            pie_fig = pie_chart(combined_df, selected_contents, color_map)
            line_fig = line_chart(combined_df, selected_times, color_map)

            col1, col2 = st.columns([2, 1])  # Adjust column width ratios as needed
            with col1:
                st.pyplot(pie_fig)
            with col2:
                st.pyplot(line_fig)




if __name__ == "__main__":
    main()
