 # Encoding for the categorical features

********************

 'What is the Overall material and finish quality?',
 'What is the Quality of the basement finished area?',
 'Where are the physical locations within Ames city limits?',
 'Where is the location of the Garage?',
 'What is the condition of the sale?',
 'Does the house have walkout or garden-level basement walls?'

********************

OverallQual: Rates the overall material and finish of the house

       10     Very Excellent
       9      Excellent
       8      Very Good
       7      Good
       6      Above Average
       5      Average
       4      Below Average
       3      Fair
       2      Poor
       1      Very Poor
********************

BsmtFinType1: Rating of basement finished area

(These will be encoded with 1 to 7 in the slider of the APP)

       GLQ    Good Living Quarters
       ALQ    Average Living Quarters
       BLQ    Below Average Living Quarters      
       Rec    Average Rec Room
       LwQ    Low Quality
       Unf    Unfinshed
       NA     No Basement

********************

Neighborhood: Physical locations within Ames city limits

(These will be encoded with 1 to 25 in the slider of the APP)

       Blmngtn	Bloomington Heights
       Blueste	Bluestem
       BrDale	       Briardale
       BrkSide	Brookside
       ClearCr	Clear Creek
       CollgCr	College Creek
       Crawfor	Crawford
       Edwards	Edwards
       Gilbert	Gilbert
       IDOTRR	       Iowa DOT and Rail Road
       MeadowV	Meadow Village
       Mitchel	Mitchell
       Names	       North Ames
       NoRidge	Northridge
       NPkVill	Northpark Villa
       NridgHt	Northridge Heights
       NWAmes	       Northwest Ames
       OldTown	Old Town
       SWISU	       South & West of Iowa State University
       Sawyer	       Sawyer
       SawyerW	Sawyer West
       Somerst	Somerset
       StoneBr	Stone Brook
       Timber	       Timberland
       Veenker	Veenker

********************

GarageType: Garage location

(These will be encoded with 1 to 7 in the slider of the APP)
              
       2Types        More than one type of garage
       Attchd        Attached to home
       Basment       Basement Garage
       BuiltIn       Built-In (Garage part of house - typically has room above garage)
       CarPort       Car Port
       Detchd        Detached from home
       NA            No Garage
              
********************

SaleCondition: Condition of sale

(These will be encoded with 1 to 6 in the slider of the APP)

       Normal Normal Sale
       Abnorml       Abnormal Sale -  trade, foreclosure, short sale
       AdjLand       Adjoining Land Purchase
       Alloca Allocation - two linked properties with separate deeds, typically condo with a garage unit 
       Family Sale between family members
       Partial       Home was not completed when last assessed (associated with New Homes)

********************

BsmtExposure: Refers to walkout or garden level walls

(These will be encoded with 1 to 5 in the slider of the APP)

       Gd     Good Exposure
       Av     Average Exposure (split levels or foyers typically score average or above)   
       Mn     Mimimum Exposure
       No     No Exposure
       NA     No Basement

