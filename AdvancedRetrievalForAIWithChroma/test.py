text=f"""# Insights from Comparing Formycon AG's Asset and Liability Composition in their Half-Year Report 2023

## Overview
Formycon AG is a German biotechnology company specializing in the development of antibodies and antibody-based therapies. In this analysis, we will focus on the insights that can be gained from comparing Formycon AG's asset and liability composition at two reporting dates mentioned in their Half-Year Report 2023: January 1 – June 30, 2023 (Interim Period 1), and July 1 – December 31, 2022 (Fiscal Year 2022).

## Comparison of Key Assets
- **Inventories**: The decrease in inventories from €324,895 to €-393,769 indicates that the company's investment in raw materials and work-in-progress may have been higher during Interim Period 1 than Fiscal Year 2022. This could be due to increased production activities or higher R&D spending.
- **Trade and other receivables**: The decrease in trade and other receivables from €16,825,973 to €16,653,317 suggests that the company may have had more outstanding accounts receivable during Interim Period 1 than Fiscal Year 2022. This could be due to higher sales revenue or longer average collection periods.
- **Contract assets**: The decrease in contract assets from €408,952 to €-7,026 indicates that the value of Formycon AG's contractual obligations decreased during Interim Period 1 compared to Fiscal Year 2022. This could be due to the settlement or cancellation of contracts.
- **Other financial assets**: No significant change was observed in other financial assets between the two reporting periods.

## Comparison of Key Liabilities
- **Trade and other payables**: The increase in trade payables from €4,060,583 to €8,402,185 indicates that Formycon AG had higher outstanding accounts payable during Interim Period 1 than Fiscal Year 2022. This could be due to increased raw material purchases or higher operating expenses.
- **Contract liabilities**: The increase in contract liabilities from €1,336,984 to €2,672,510 suggests that the company entered into more contracts during Interim Period 1 than Fiscal Year 2022, or the value of existing contracts increased.
- **Other liabilities**: The decrease in other liabilities from €-267,958 to €-403,338 indicates that Formycon AG had less outstanding obligations to third parties during Interim Period 1 compared to Fiscal Year 2022. This could be due to the settlement or repayment of debts.

## Conclusion
Comparing Formycon AG's asset and liability composition at two reporting dates provides valuable insights into the company's financial performance during different periods. By analyzing changes in key assets and liabilities, we can identify trends related to production activities, sales revenue, R&D spending, contractual obligations, and operating expenses. This information is useful for investors and analysts seeking to understand Formycon AG's financial health and growth prospects.%   """


text2=f"""
# Analysis of Formycon AG's

Formycon AG's Half-Year Report 2023 provides insights into the company's asset and liability composition at two reporting dates: June 30, 2023, and December 31, 2022. Let's examine the changes in each category.

## Assets

### Non-current assets

Formycon AG reported an increase of €76,354 ($84,879) in non-current assets between the two reporting dates. This increase can be attributed to:
1. **Goodwill**: No change.
2. **Other intangible assets**: An increase of €118,221 ($130,965).
3. **Right-of-use (ROU) assets**: An increase of €142.
4. **Property, plant and equipment**: A decrease of €227.
5. **Investment participations at equity**: A decrease of €16,162 ($18,342).
6. **Financial assets**: An increase of €1,150 ($1,285).

### Current assets

The total current assets increased by €52,893 ($58,978) between June 30, 2023, and December 31, 2022. The growth was mainly driven by:
1. **Inventories**: An increase of €394.
2. **Trade and other receivables**: A significant surge of €16,556 ($18,766).
3. **Contract assets**: An increase of €6,801 ($7,702).
4. **Prepayments and other assets**: An insignificant increase of €1,957 ($2,187).
5. **Cash and cash equivalents**: A substantial rise of €26,312 ($29,213).

## Liabilities

### Non-current liabilities

Formycon AG reported a decrease in total non-current liabilities by €42,075 ($47,865) between the two reporting dates. This decline can be attributed to:
1. **Lease liabilities**: A significant decrease of €39,246 ($44,534).
2. **Other liabilities**: An insignificant decrease of €2,829 ($3,231).

### Current liabilities

Total current liabilities increased by €7,081 ($8,072) between June 30, 2023, and December 31, 2022. This increase can be attributed to:
1. **Accounts payable**: An increase of €2,194 ($2,469).
2. **Accrued liabilities**: An insignificant increase of €252 ($283).
3. **Taxes payable**: A significant surge of €4,635 ($5,270).

In summary, the comparison of Formycon AG's asset and liability composition at June 30, 2023, and December 31, 2022, reveals an increase in total assets by €129,448 ($145,837) and a decrease in total liabilities by €35,000 ($39,603). The major contributors to the growth in assets were other intangible assets, trade and other receivables, contract assets, cash and cash equivalents, and financial assets. The significant decrease in non-current liabilities was mainly due to a decrease in lease liabilities. Current liabilities also grew, primarily driven by an increase in taxes payable.Presentation created successfully with improved formatting."""

un_process_results = []
startIndex = 0

while True:
    startIndex = text2.find("#", startIndex)
    
    # If no more '#' found, break out of the loop
    if startIndex == -1:
        break
    
    # Check if the next character is not another '#'
    if text2[startIndex+1] != "#":
        nextIndex = text2.find("#", startIndex + 1)
        # Ensure we don't include the next '#' in the substring
        while nextIndex != -1 and text2[nextIndex+1] == "#":
            nextIndex = text2.find("#", nextIndex + 1)
        
        if nextIndex == -1:
            segment = text2[startIndex:]
        else:
            segment = text2[startIndex:nextIndex]
        
        un_process_results.append(segment.strip())
        
    # Move startIndex forward to avoid re-finding the same '#'
    startIndex += 1

print("Unprocessed results:", (un_process_results))
# print("un_process_results : ",un_process_results)

for line in un_process_results:
    print("\n",line)