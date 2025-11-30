import matplotlib
matplotlib.use('Agg') 
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import platform
import subprocess
import datetime as dt 
import numpy as np 
def load_data():
    csv_file = 'canteen_sales_data.csv'
    df = pd.read_csv(csv_file)
    df['Date'] = pd.to_datetime(df['Date'])
    items = ['Veg_Patties', 'Pizza_Slice', 'Veg_Sandwich', 'Chicken_Burger', 'Fruit_Juice']
    if 'Total_Daily_Sales' not in df.columns:
        valid_items = [i for i in items if i in df.columns]
        df['Total_Daily_Sales'] = df[valid_items].sum(axis=1)
        
    return df, items

def show_overall_dashboard(df, items):
    """Generates and displays the 4-chart analysis dashboard with improved styling."""
    print("\n📊 Generating Enhanced Sales Report... now with more color and clarity!")
    item_totals = df[items].sum().sort_values(ascending=False)
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['Day_of_Week'] = df['Date'].dt.day_name() 
    day_avg = df.groupby('Day_of_Week')['Total_Daily_Sales'].mean().reindex(day_order).fillna(0)
    
    event_avg = df.groupby('Event')['Total_Daily_Sales'].mean().sort_values()
    
    overall_avg = df['Total_Daily_Sales'].mean() 

    fig, axs = plt.subplots(2, 2, figsize=(16, 12), facecolor='#f4f4f4')
    plt.subplots_adjust(hspace=0.5, wspace=0.3, top=0.9)
    fig.suptitle('Canteen Sales Analytics Dashboard - Performance Overview', fontsize=20, color='#2c3e50', fontweight='bold')

    axs[0, 0].plot(df['Date'], df['Total_Daily_Sales'], color='#3498db', linewidth=3, marker='o', markersize=5, markevery=5, markerfacecolor='white', markeredgecolor='#2c3e50')
    axs[0, 0].set_title('Daily Sales Volume Trend', fontweight='bold', color='#2c3e50')
    axs[0, 0].set_ylabel('Total Items Sold')
    axs[0, 0].set_xlabel('Date')

    axs[0, 0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
    axs[0, 0].tick_params(axis='x', rotation=45)
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.6)
    axs[0, 0].tick_params(axis='both', labelsize=10)


    item_colors = ['#1abc9c', '#e67e22', '#3498db', '#f1c40f', '#9b59b6']
    bars2 = axs[0, 1].bar(item_totals.index, item_totals.values, color=item_colors[:len(item_totals)])
    axs[0, 1].set_title('Top Selling Food Items (Total Quantity)', fontweight='bold', color='#2c3e50')
    axs[0, 1].set_ylabel('Quantity Sold')
    axs[0, 1].tick_params(axis='x', rotation=15)
    

    for bar in bars2:
        yval = bar.get_height()
        axs[0, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 10, int(yval), ha='center', va='bottom', fontsize=10)

    day_colors = ['#3498db'] * 7 
    day_colors[day_order.index('Friday')] = '#e67e22' 
    if 'Saturday' in day_order:
        day_colors[day_order.index('Saturday')] = '#9b59b6' 
    if 'Sunday' in day_order:
        day_colors[day_order.index('Sunday')] = '#e74c3c' 

    bars3 = axs[1, 0].bar(day_avg.index, day_avg.values, color=day_colors)
    axs[1, 0].set_title('Average Sales by Day of Week', fontweight='bold', color='#2c3e50')
    axs[1, 0].set_ylabel('Avg Items Sold')
    axs[1, 0].tick_params(axis='x', rotation=45)
    axs[1, 0].grid(axis='y', linestyle='--', alpha=0.6)
    
    axs[1, 0].axhline(overall_avg, color='gray', linestyle=':', linewidth=2, label=f'Overall Avg ({overall_avg:.0f})')
    axs[1, 0].legend()

    event_avg_values = event_avg.values
    
    event_colors = [
        '#2ecc71' if x > overall_avg * 1.05 else  
        ('#e74c3c' if x < overall_avg * 0.90 else '#3498db') 
        for x in event_avg_values
    ]
    
    axs[1, 1].barh(event_avg.index, event_avg.values, color=event_colors)
    axs[1, 1].set_title('Impact of Special Events on Sales Volume', fontweight='bold', color='#2c3e50')
    axs[1, 1].set_xlabel('Avg Items Sold per Day')
    axs[1, 1].axvline(overall_avg, color='gray', linestyle=':', linewidth=2, label=f'Overall Avg ({overall_avg:.0f})')
    axs[1, 1].legend()

    filename = "canteen_dashboard.png"
    plt.savefig(filename)
    plt.close(fig) 
    print(f"✅ Dashboard saved as '{filename}'.")   
    print("🖥️  Opening the image for you...")
    try:
        if platform.system() == "Windows":
            os.startfile(filename)
        elif platform.system() == "Darwin":  #
            subprocess.call(["open", filename])
        else:  # Linux
            subprocess.call(["xdg-open", filename])
    except Exception as e:
        print(f"⚠️  Could not auto-open. Please find '{filename}' in your folder and open it manually.")

def predict_sales(df, items):
    """
    Predicts item-wise sales for the next 7 days using seasonal averages 
    and visualizes the forecast with improved aesthetics and in-bar item counts.
    """
    print("\n🔮 Initiating Predictive Sales Model...")

    print("   - Calculating historical seasonality...")
    df['Day_of_Week'] = df['Date'].dt.day_name()
    seasonal_avg = df.groupby('Day_of_Week')[items].mean()
    last_date = df['Date'].max()
    next_day_date = last_date + pd.Timedelta(days=1)
    
    prediction_dates = [next_day_date + pd.Timedelta(days=i) for i in range(7)]
    prediction_df = pd.DataFrame({'Date': prediction_dates})
    prediction_df['Day_of_Week'] = prediction_df['Date'].dt.day_name()
    
    predicted_sales = []
    for index, row in prediction_df.iterrows():
        day_name = row['Day_of_Week']
        prediction_row = {'Date': row['Date'].date()}
        
        if day_name in seasonal_avg.index:
            for item in items:
                prediction_row[item] = seasonal_avg.loc[day_name, item]
        else: 
            for item in items:
                prediction_row[item] = 0
                
        predicted_sales.append(prediction_row)
        
    predicted_df = pd.DataFrame(predicted_sales).set_index('Date')
    predicted_df = predicted_df.round(0).astype(int) 
    predicted_df['Total'] = predicted_df.sum(axis=1)

    print(f"   - Prediction complete, next day is {next_day_date.strftime('%A, %B %d')}.")
    next_day_pred = predicted_df.iloc[0]
    print("\n--- 📅 NEXT DAY PREDICTION SUMMARY ---")
    print(f"Target Date: {next_day_date.strftime('%A, %B %d, %Y')}")
    print("---------------------------------------")
    for item in items:
        print(f"  {item.replace('_', ' '):<15}: {next_day_pred[item]:>4} units")
    print("---------------------------------------")
    print(f"  TOTAL PREDICTED: {next_day_pred['Total']:>4} units")
    print("---------------------------------------")

    fig, axs = plt.subplots(1, 2, figsize=(18, 8), facecolor='#ffffff') 
    
    plt.subplots_adjust(wspace=0.3, top=0.88, bottom=0.15, right=0.85)
    
    fig.suptitle(f'Canteen Weekly Sales Forecast ({next_day_date.strftime("%B %d, %Y")} to {prediction_dates[-1].strftime("%B %d, %Y")})', 
                 fontsize=20, color='#2c3e50', fontweight='bold', y=0.98)

    ITEM_PALETTE = ['#27ae60', '#f39c12', '#3498db', '#e74c3c', '#9b59b6']
    
    day_order_pred = [date.strftime('%a') for date in prediction_dates]
    
 
    predicted_df[items].plot(kind='bar', stacked=True, color=ITEM_PALETTE[:len(items)], ax=axs[0], width=0.8)

    axs[0].set_title('Daily Item Mix and Total Volume Forecast (with Piece Count)', 
                     fontweight='bold', color='#34495e', fontsize=14)
    axs[0].set_ylabel('Total Units Predicted', fontsize=12)
    axs[0].set_xlabel('Day of Week', fontsize=12)
    axs[0].set_xticklabels(day_order_pred, rotation=0, fontsize=11)
    
    total_sales = predicted_df['Total'].values
    x_positions = np.arange(len(day_order_pred))
    for x, total in zip(x_positions, total_sales):
        axs[0].text(x, total + 10, str(total), ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c3e50')
       
    for i, item in enumerate(items):
        item_values = predicted_df[item].values
        
        if i == 0:
            bottoms = np.zeros(len(x_positions))
        else:
            bottoms = predicted_df[items[:i]].sum(axis=1).values

        tops = bottoms + item_values
        centers = (bottoms + tops) / 2 

        for x, value, center in zip(x_positions, item_values, centers):
            if value > 5: 
                axs[0].text(x, center, str(value), 
                            ha='center', va='center', 
                            color='white', fontsize=9, fontweight='semibold')
    axs[0].legend(title='Item', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    axs[0].grid(axis='y', linestyle='-', alpha=0.3)
    
    top_3_items = predicted_df[items].sum().sort_values(ascending=False).head(3).index.tolist()
    
    line_palette = ['#e74c3c', '#2980b9', '#16a085'] 
    markers = ['o', 's', '^']
    for i, item in enumerate(top_3_items):
        axs[1].plot(day_order_pred, predicted_df[item], 
                    marker=markers[i], markersize=7, linestyle='--',
                    label=item.replace('_', ' '), color=line_palette[i], 
                    linewidth=2.5)
    axs[1].set_title('Weekly Trend of Top 3 Items', 
                     fontweight='bold', color='#34495e', fontsize=14)
    axs[1].set_ylabel('Units Predicted', fontsize=12)
    axs[1].set_xlabel('Day of Week', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend(title='Item', frameon=False, loc='upper left')
    axs[1].tick_params(axis='x', rotation=0)

    # Save and Open
    filename = "sales_forecast_item_pieces.png"
    plt.savefig(filename)
    plt.close(fig) 
    print(f"\n✅ Forecast dashboard saved as '{filename}'.")
    
    print("🖥️  Opening the image for you...")
    try:
        if platform.system() == "Windows":
            os.startfile(filename)
        elif platform.system() == "Darwin":  
            subprocess.call(["open", filename])
        else:  # Linux
            subprocess.call(["xdg-open", filename])
    except Exception as e:
        print(f"⚠️  Could not auto-open. Please find '{filename}' in your folder and open it manually.")
def get_valid_input(prompt, max_option):
    """Safely gets a number from the user."""
    while True:
        try:
            value = int(input(prompt))
            if 1 <= value <= max_option:
                return value
            print(f"❌ Invalid choice. Please enter a number between 1 and {max_option}.")
        except ValueError:
            print("❌ That's not a number! Please try again.")

def print_header():
    print("\n" + "="*40)
    print("   🍱 SCHOOL CANTEEN ANALYTICS HUB 🍱   ")
    print("="*40)

def main():
    df, items = load_data()
    while True:
        print_header()
        print("1. 📈 View Sales Reports")
        print("2. 🔮 Predict Future Sales") 
        print("3. 🚪 Exit")
        choice = get_valid_input("\n👉 Enter your option: ", 3)
        if choice == 1:
            print("\n--- Sales Reports Menu ---")
            print("1. Overall Sales Dashboard")
            print("2. Specific Item Report (Text Only)")
            print("3. Back to Main Menu")
            
            sub_choice = get_valid_input("👉 Select Report Type: ", 3)

            if sub_choice == 1:
                show_overall_dashboard(df, items)

            elif sub_choice == 2:
                print("\nWhich item?")
                for idx, item in enumerate(items, 1):
                    print(f"{idx}. {item}")
                item_idx = get_valid_input("Select item number: ", len(items))
                selected_item = items[item_idx-1]
                total = df[selected_item].sum()
                avg = df[selected_item].mean()
                print(f"\n🍔 Report for {selected_item}:")
                print(f"   - Total Sold: {total}")
                print(f"   - Daily Average: {avg:.1f}")
                input("\nPress Enter to continue...")
        elif choice == 2:
            predict_sales(df, items)
            input("\nPress Enter to continue...")

        elif choice == 3:
            print("\n👋 Goodbye! Have a great day!")
            break

        else:
            print("\n🚧 Feature under construction! Try 'Sales Reports' or 'Predict Future Sales' for now.")
            time.sleep(1)
main()