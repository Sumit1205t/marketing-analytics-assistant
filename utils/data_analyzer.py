def analyze_data(self, df: pd.DataFrame, analysis_id: str = None) -> Dict[str, Any]:
    """Analyze data with configurable metrics."""
    try:
        df = self.preprocess_data(df)
        analysis = {
            'overview': {},
            'trends': {},
            'segments': {},
            'recommendations': []
        }
        
        # Filter for specific analysis if provided
        if analysis_id:
            df = df[df['analysis_id'] == analysis_id]
        
        # Calculate metrics based on configuration
        analysis['overview'] = self._calculate_overview_metrics(df)
        
        # Calculate time-based trends if temporal data exists
        date_col = next((col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])), None)
        if date_col:
            analysis['trends'] = self._calculate_trends(df, date_col)
        
        # Analyze segments
        analysis['segments'] = self._analyze_segments(df)
        
        # Generate insights
        analysis['insights'] = self._generate_insights(analysis)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")
        return {} 